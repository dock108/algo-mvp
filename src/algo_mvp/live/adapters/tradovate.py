import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import httpx
import websockets
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from algo_mvp.live.broker_adapter_base import BrokerAdapterBase
from algo_mvp.live.models import Fill, Order, Position  # Assuming these exist

# Configure logging
logger = logging.getLogger(__name__)

# Environment Variables
TRADOVATE_CLIENT_ID = os.getenv("TRADOVATE_CLIENT_ID")
TRADOVATE_USERNAME = os.getenv("TRADOVATE_USERNAME")
TRADOVATE_PASSWORD = os.getenv("TRADOVATE_PASSWORD")
TRADOVATE_BASE_URL = os.getenv("TRADOVATE_BASE_URL", "https://demo.tradovateapi.com/v1")
TRADOVATE_WS_URL = os.getenv(
    "TRADOVATE_WS_URL", "wss://demo.tradovateapi.com/v1/websocket"
)

# Constants
HEARTBEAT_INTERVAL = 30  # seconds
MAX_REQUESTS_PER_SECOND = 30
RECONNECT_DELAY = 5  # seconds
POLL_INTERVAL = 60  # seconds for cash/positions


class TradovateAccessTokenResponse(BaseModel):
    accessToken: str
    userId: int
    userStatus: str
    expirationTime: datetime
    passwordExpirationTime: datetime | None = None
    name: str
    hasLive: bool

    @property
    def is_expired(self) -> bool:
        # Add a small buffer to account for potential clock skew or network latency
        return datetime.now(timezone.utc) >= (
            self.expirationTime - timedelta(seconds=60)
        )


class TradovateBrokerAdapter(BrokerAdapterBase):
    def __init__(
        self,
        runner,
        client_id: str | None = None,
        username: str | None = None,
        password: str | None = None,
        base_url: str | None = None,
        ws_url: str | None = None,
    ):
        super().__init__(runner)
        self.client_id = client_id or TRADOVATE_CLIENT_ID
        self.username = username or TRADOVATE_USERNAME
        self.password = password or TRADOVATE_PASSWORD
        self.base_url = base_url or TRADOVATE_BASE_URL
        self.ws_url = ws_url or TRADOVATE_WS_URL

        if not all([self.client_id, self.username, self.password]):
            raise ValueError(
                "Tradovate credentials (client_id, username, password) must be provided either as arguments or environment variables."
            )

        self.http_client = httpx.AsyncClient(base_url=self.base_url)
        self.access_token_details: TradovateAccessTokenResponse | None = None
        self.ws_connection = None
        self.ws_listener_task = None
        self.heartbeat_task = None
        self.is_connecting = False
        self.is_disconnecting = False
        self.last_request_time = 0
        self.request_count_this_second = 0
        self._stop_event = asyncio.Event()

        # For get_cash / get_positions polling
        self._account_id: int | None = None
        self._cash: dict[str, float] = {}
        self._positions: list[Position] = []
        self._data_poller_task = None

        logger.info("TradovateBrokerAdapter initialized.")

    async def _rate_limit_request(self):
        """Ensures compliance with rate limits."""
        current_time = time.time()
        if current_time - self.last_request_time < 1:
            self.request_count_this_second += 1
            if self.request_count_this_second >= MAX_REQUESTS_PER_SECOND:
                logger.warning("Rate-limited, backing off.")
                await asyncio.sleep(1 - (current_time - self.last_request_time))
                self.request_count_this_second = 0  # Reset after waiting
                self.last_request_time = time.time()  # Update last request time
        else:
            self.last_request_time = current_time
            self.request_count_this_second = 1

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=0.5, max=2))
    async def _make_request(
        self, method: str, endpoint: str, **kwargs
    ) -> httpx.Response:
        """Makes an HTTP request with rate limiting and retries."""
        if self.is_disconnecting:
            logger.info(f"Skipping request to {endpoint} as adapter is disconnecting.")
            # Or raise an exception if preferred for immediate termination of calling logic
            raise asyncio.CancelledError("Adapter is disconnecting.")

        await self._rate_limit_request()

        headers = kwargs.pop("headers", {})
        if self.access_token_details and not self.access_token_details.is_expired:
            headers["Authorization"] = f"Bearer {self.access_token_details.accessToken}"
        else:
            # If no token or expired, and not an auth request, attempt to refresh
            if "/auth/" not in endpoint:
                logger.info(
                    "Access token is missing or expired. Attempting to refresh."
                )
                await self._get_access_token()  # This will update self.access_token_details
                if self.access_token_details:  # Check again after refresh attempt
                    headers["Authorization"] = (
                        f"Bearer {self.access_token_details.accessToken}"
                    )
                else:
                    logger.error("Failed to refresh access token. Request may fail.")
                    # Potentially raise an error here if a token is strictly required

        logger.debug(
            f"Making {method} request to {endpoint} with headers: {list(headers.keys())} and params/data: {kwargs.get('params') or kwargs.get('json') or kwargs.get('data')}"
        )  # Used list(headers.keys()) for clarity
        response = await self.http_client.request(
            method, endpoint, headers=headers, **kwargs
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        return response

    async def _get_access_token(self) -> None:
        """Authenticates and retrieves an access token."""
        logger.info("Requesting new access token...")
        try:
            data = {
                "name": self.username,
                "password": self.password,
                "appId": "Sample App",  # As per Tradovate API docs
                "appVersion": "1.0",
                "cid": self.client_id,  # This should be your device ID or unique client ID from Tradovate
                "deviceId": "algo_mvp_device_001",  # This needs to be unique per session/device typically
                "deviceVersion": "1.0",
                "secType": "token",  # For OAuth-like token request
            }
            # This request should not use _make_request to avoid circular dependency on token
            await self._rate_limit_request()
            response = await self.http_client.post(
                "/auth/accesstokenrequest", json=data
            )
            response.raise_for_status()
            token_data = response.json()
            self.access_token_details = TradovateAccessTokenResponse(**token_data)
            # Store account ID if available from token response or subsequent calls
            # Example: self._account_id = self.access_token_details.userId (if userId is accountId)
            logger.info(
                f"Successfully obtained access token, expires at {self.access_token_details.expirationTime}"
            )

            # After getting a new token, if WS was connected, it might need re-authentication or restart
            if self.ws_connection and not self.ws_connection.closed:
                logger.info(
                    "Access token refreshed. Re-authenticating WebSocket or reconnecting if necessary."
                )
                # Tradovate WS typically requires sending the token with the initial 'authorize' message.
                # If the WS connection is kept alive, you might need to re-authorize.
                # For simplicity, we might opt to close and reopen the WS connection on token expiry.
                # Or send an explicit re-auth message if supported by Tradovate's WS protocol.
                # For now, we'll rely on the connect() method to handle WS setup after token acquisition.
                # Or, if a heartbeat failure leads to reconnect, it will use the new token.

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Failed to get access token: {e.response.status_code} - {e.response.text}"
            )
            self.access_token_details = None  # Ensure token is None on failure
            raise  # Re-raise the exception to be handled by the caller or retry mechanism
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during access token retrieval: {e}"
            )
            self.access_token_details = None  # Ensure token is None on failure
            raise

    async def connect(self):
        if self.is_connecting or (
            self.access_token_details
            and not self.access_token_details.is_expired
            and self.ws_connection
            and not self.ws_connection.closed
        ):
            logger.info("Connection attempt skipped: already connecting or connected.")
            return

        self.is_connecting = True
        self._stop_event.clear()  # Clear stop event for new connection attempt
        self.is_disconnecting = False  # Ensure this is false when attempting to connect

        try:
            logger.info("Connecting to Tradovate...")
            if not self.access_token_details or self.access_token_details.is_expired:
                await self._get_access_token()

            if not self.access_token_details:
                logger.error("Failed to connect: Could not obtain access token.")
                self.is_connecting = False
                return

            # Establish WebSocket connection
            await self._ws_connect()

            # Start heartbeat
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
            self.heartbeat_task = asyncio.create_task(self._send_ws_heartbeat())

            # Start listener for WebSocket messages
            if self.ws_listener_task:
                self.ws_listener_task.cancel()
            self.ws_listener_task = asyncio.create_task(self._ws_listen())

            # Start data poller for cash and positions
            if self._data_poller_task:
                self._data_poller_task.cancel()
            self._data_poller_task = asyncio.create_task(self._poll_data())

            logger.info("TradovateBrokerAdapter connected successfully.")
            if self.runner:  # Notify runner if available
                asyncio.create_task(
                    self.runner.on_broker_event(
                        {"type": "connect", "status": "success"}
                    )
                )

        except Exception as e:
            logger.error(f"Failed to connect to Tradovate: {e}", exc_info=True)
            if self.runner:  # Notify runner of failure
                asyncio.create_task(
                    self.runner.on_broker_event(
                        {"type": "connect", "status": "failure", "error": str(e)}
                    )
                )
            # Consider a brief sleep before allowing reconnect attempts to prevent rapid failing loops.
        finally:
            self.is_connecting = False

    async def _ws_connect(self):
        if not self.access_token_details or self.access_token_details.is_expired:
            logger.error(
                "Cannot connect WebSocket: Access token is missing or expired."
            )
            # Attempt to refresh token before failing
            await self._get_access_token()
            if not self.access_token_details or self.access_token_details.is_expired:
                raise ConnectionError(
                    "WebSocket connection failed: No valid access token."
                )

        try:
            logger.debug(f"Attempting to connect to WebSocket: {self.ws_url}")
            self.ws_connection = await websockets.connect(
                self.ws_url, ssl=True, ping_interval=None
            )  # Explicitly disable auto ping
            logger.info(f"WebSocket connection established to {self.ws_url}")

            auth_message = f"authorize\n0\n\n{self.access_token_details.accessToken}"
            await self.ws_connection.send(auth_message)
            logger.debug(f"Sent WebSocket authorization message.")

            # Handle initial 'o' frame (connection open)
            initial_response = await self.ws_connection.recv()
            logger.debug(f"WebSocket initial response: {initial_response}")
            if initial_response != "o":  # 'o' signifies connection opened by server
                raise ConnectionError(
                    f"WebSocket connection failed: Expected 'o' frame, got {initial_response}"
                )

            auth_confirm_response = (
                await self.ws_connection.recv()
            )  # Should be auth confirmation
            logger.info(
                f"WebSocket authorization confirmation: {auth_confirm_response}"
            )
            if not auth_confirm_response.startswith("a["):  # 'a' for array message type
                raise ConnectionError(
                    f"WebSocket authorization failed: Unexpected response {auth_confirm_response}"
                )
            # Further parsing of auth_confirm_response can be done here if needed

            # Subscribe to order and fill reports
            # deviceId = self.access_token_details.deviceId # Or the one used in auth. Needs to be consistent.
            # The deviceId in subscribe should match the one used for accessTokenRequest if that's how Tradovate links them.
            # For now, using a fixed or configured deviceId for subscribe.
            # Let's assume the API doesn't require deviceId for these subscriptions explicitly in this manner.
            # Subscriptions are typically simple strings for the entity type.
            # Tradovate's actual subscription mechanism is via user/account properties or specific endpoints.
            # The prompt asks for order:report and fill:report. This usually means subscribing to account-level events.
            # This part needs to match Tradovate's specific WebSocket API for subscribing to reports.
            # The common way is `GET /user/syncrequest` with body like: {"users": [userId], "accounts": [accountId]}
            # Then events will stream. Or, simpler subscriptions if available.
            # For now, we assume a simple subscription model if one exists, or that events stream post-auth.
            # The prompt: "subscribe to order:report + fill:report"
            # This might be done via /user/syncrequest or another mechanism.
            # Let's assume for now that authorization implies subscription to relevant events.
            # If specific subscribe messages are needed, they would be sent here.
            # Example: await self.ws_connection.send('subscribe\n0\n\norder:report')
            #          await self.ws_connection.send('subscribe\n0\n\nfill:report')
            # This is a placeholder for the actual subscription mechanism.
            # Tradovate often uses a "sync request" to start data flow for accounts/users.
            # Let's ensure the accountId is known.
            if not self._account_id:
                await self._get_initial_account_data()  # Fetch account list to get an ID

            if self._account_id:
                # Example sync request: {"url":"user/syncrequest","body":{"users":[<userId>],"accounts":[<accountId>]}}
                # This is a complex object that gets sent typically.
                # Let's simplify and assume the LiveRunner.on_trade is for fills, and we monitor order updates internally.
                # Tradovate sends "event" messages which are arrays.
                # Frame type 'a' (array of JSON objects)
                # Each object has 'e' (event type, e.g., "props", "fill", "order") and 'd' (data)
                logger.info(
                    "WebSocket connected and authorized. Listening for events..."
                )

        except websockets.exceptions.InvalidURI:
            logger.error(f"Invalid WebSocket URI: {self.ws_url}")
            raise
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.ws_connection = None  # Ensure connection is None on failure
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error during WebSocket connection: {e}", exc_info=True
            )
            if self.ws_connection and not self.ws_connection.closed:
                await self.ws_connection.close()
            self.ws_connection = None
            raise

    async def _send_ws_heartbeat(self):
        """Sends heartbeat messages over WebSocket."""
        while not self._stop_event.is_set():
            try:
                if self.ws_connection and not self.ws_connection.closed:
                    # Tradovate WS heartbeat is just sending 'h'
                    # Or, as per some docs, it's just keeping the connection alive,
                    # and the server sends 'h' which client should respond to.
                    # The prompt: "start heartbeat timer (30 s)" -> implies client-side initiated or server responded.
                    # Let's assume client sends a simple heartbeat message.
                    # If Tradovate expects client to respond to server 'h', this needs adjustment.
                    # Sending a null message or a specific heartbeat command might be required.
                    # Simplest keep-alive if no specific client-send heartbeat:
                    # await self.ws_connection.ensure_open()
                    # Or, if a message is needed:
                    await self.ws_connection.send(
                        "[]"
                    )  # Empty array, often used as a benign keep-alive/heartbeat
                    logger.debug("Sent WebSocket heartbeat.")
                else:
                    logger.warning(
                        "WebSocket not connected, cannot send heartbeat. Attempting to reconnect..."
                    )
                    asyncio.create_task(self.connect())  # Attempt to reconnect
                    await asyncio.sleep(
                        HEARTBEAT_INTERVAL
                    )  # Wait before next attempt if reconnect fails quickly
                    continue  # Skip the rest of the loop and re-evaluate connection status
                await asyncio.sleep(HEARTBEAT_INTERVAL)
            except websockets.exceptions.ConnectionClosed:
                logger.warning(
                    "WebSocket connection closed while sending heartbeat. Attempting to reconnect."
                )
                self.ws_connection = None
                if (
                    not self.is_disconnecting
                ):  # Avoid reconnect if we are intentionally disconnecting
                    asyncio.create_task(self.connect())  # Non-blocking reconnect
                    await asyncio.sleep(
                        RECONNECT_DELAY
                    )  # Wait a bit before trying to send next heartbeat
            except asyncio.CancelledError:
                logger.info("Heartbeat task cancelled.")
                break
            except Exception as e:
                logger.error(
                    f"Error in WebSocket heartbeat: {e}. Attempting to reconnect."
                )
                self.ws_connection = None
                if not self.is_disconnecting:
                    asyncio.create_task(self.connect())
                    await asyncio.sleep(RECONNECT_DELAY)

    async def _ws_listen(self):
        """Listens for messages from WebSocket and handles them."""
        if not self.ws_connection:
            logger.error("Cannot listen: WebSocket not connected.")
            return

        logger.info("Starting WebSocket listener...")
        try:
            while not self._stop_event.is_set():
                if not self.ws_connection or self.ws_connection.closed:
                    logger.warning(
                        "WebSocket connection lost in listener. Attempting reconnect."
                    )
                    if not self.is_disconnecting:
                        await self.connect()  # Re-establish connection
                        if (
                            not self.ws_connection or self.ws_connection.closed
                        ):  # If connect failed
                            await asyncio.sleep(
                                RECONNECT_DELAY
                            )  # Wait before retrying listen loop
                            continue
                    else:  # if is_disconnecting
                        break

                try:
                    message = await asyncio.wait_for(
                        self.ws_connection.recv(), timeout=HEARTBEAT_INTERVAL + 5
                    )
                    # Tradovate WS messages are typically strings.
                    # 'o' - open frame (already handled in _ws_connect)
                    # 'h' - heartbeat from server (client should respond if required by API, or just ignore)
                    # 'c' - close frame
                    # 'a' - array of JSON objects (most data like orders, fills, positions, props)

                    if message == "h":  # Server heartbeat
                        logger.debug("Received server heartbeat.")
                        # Respond if necessary, e.g., await self.ws_connection.send("h_response_if_needed")
                        continue

                    if message.startswith("a"):  # Array message
                        try:
                            data_str = message[1:]  # Remove 'a' prefix
                            events = json.loads(data_str)
                            for event_obj in events:
                                if not isinstance(event_obj, dict):
                                    logger.warning(
                                        f"Received non-dict event in array: {event_obj}"
                                    )
                                    continue

                                event_type = event_obj.get("e")
                                event_data = event_obj.get("d")

                                if (
                                    event_type == "props"
                                ):  # Account properties, might contain cash, etc.
                                    # logger.debug(f"Props event: {event_data}")
                                    # This is where account updates (like cash balance from streaming) could be parsed
                                    # For example, if 'cash' is a prop:
                                    if "cash" in event_data:
                                        self._cash = float(event_data["cash"])
                                        logger.info(
                                            f"Cash updated via WS: {self._cash}"
                                        )
                                    # Update other relevant account details if present
                                    pass
                                elif (
                                    event_type == "fill" and self.runner
                                ):  # fill:report
                                    # This is a fill event. We need to map it to our internal Fill model
                                    # and call self.runner.on_trade()
                                    # Example fill structure (highly dependent on Tradovate's actual data)
                                    # { "e": "fill", "d": { "tradeId": 12345, "orderId": 67890, "contractId": 789,
                                    #                        "timestamp": "2025-05-09T10:00:00Z", "action": "Buy", "qty": 1,
                                    #                        "price": 5000, "active": true, ... } }
                                    logger.info(f"Received fill event: {event_data}")
                                    fill = self._map_tradovate_fill(event_data)
                                    if fill:
                                        asyncio.create_task(self.runner.on_trade(fill))

                                elif (
                                    event_type == "order" and self.runner
                                ):  # order:report
                                    # This is an order status update.
                                    # We might update an internal order book or log it.
                                    # Example order structure:
                                    # { "e": "order", "d": { "orderId": 67890, "orderStatus": "Filled", ... } }
                                    logger.info(f"Received order event: {event_data}")
                                    order_update = self._map_tradovate_order_update(
                                        event_data
                                    )
                                    if order_update and hasattr(
                                        self.runner, "on_order_update"
                                    ):  # If runner has such a handler
                                        asyncio.create_task(
                                            self.runner.on_order_update(order_update)
                                        )
                                    # Or just log for now if no specific runner handler
                                    # logger.debug(f"Order update: {order_update}")

                                # Handle other event types as needed
                        except json.JSONDecodeError:
                            logger.error(
                                f"Failed to decode JSON from WebSocket message: {message}"
                            )
                        except Exception as e_proc:
                            logger.error(
                                f"Error processing WebSocket event: {e_proc} - Message: {message}",
                                exc_info=True,
                            )

                    elif message.startswith("c["):  # Close frame
                        logger.info(
                            f"Received WebSocket close frame: {message}. Connection will close."
                        )
                        break  # Exit listener loop

                    else:
                        logger.debug(f"Received other WebSocket message: {message}")

                except asyncio.TimeoutError:
                    logger.debug(
                        "WebSocket recv timed out (expected if no messages). Checking connection."
                    )
                    if self.ws_connection and not self.ws_connection.closed:
                        # Send a keep-alive if connection is open but idle
                        # await self.ws_connection.send("[]") # Benign keep-alive
                        pass  # Heartbeat task should handle sending regular pings/heartbeats
                    else:  # Connection seems closed or problematic
                        logger.warning(
                            "WebSocket connection appears closed or unresponsive after timeout."
                        )
                        if not self.is_disconnecting:
                            await self.connect()  # Attempt to reconnect
                        else:
                            break  # Exit if disconnecting
                except websockets.exceptions.ConnectionClosedOK:
                    logger.info("WebSocket connection closed gracefully by server.")
                    if not self.is_disconnecting:
                        await self.connect()  # Reconnect if not intentional
                    else:
                        break
                except websockets.exceptions.ConnectionClosedError as e:
                    logger.warning(
                        f"WebSocket connection closed with error: {e}. Attempting to reconnect."
                    )
                    if not self.is_disconnecting:
                        await self.connect()
                    else:
                        break
                except (
                    websockets.exceptions.ConnectionClosed
                ) as e:  # Catchall for other closed types
                    logger.warning(
                        f"WebSocket connection closed unexpectedly: {e}. Attempting to reconnect."
                    )
                    if not self.is_disconnecting:
                        await self.connect()
                    else:
                        break

        except asyncio.CancelledError:
            logger.info("WebSocket listener task cancelled.")
        except Exception as e:
            logger.error(f"Fatal error in WebSocket listener: {e}", exc_info=True)
        finally:
            logger.info("WebSocket listener stopped.")
            if (
                self.ws_connection
                and not self.ws_connection.closed
                and not self.is_disconnecting
            ):
                logger.info(
                    "Listener stopped, but WS still open and not disconnecting. This might be an issue."
                )
            # Do not close WS connection here as it might be managed by connect/close methods

    def _map_tradovate_fill(self, fill_data: dict) -> Fill | None:
        """Maps Tradovate fill data to the internal Fill model."""
        try:
            # This mapping is highly speculative and needs to be adjusted based on actual Tradovate API response
            # Required fields for Fill: id, order_id, symbol, qty, price, side, timestamp
            # Assuming 'contractId' or similar might map to a symbol lookup
            # For now, we'll need a placeholder for symbol if not directly available.

            # We need a way to get the symbol. Tradovate uses `contractId`.
            # This would typically require a lookup or that the fill data includes the symbol string.
            # Let's assume `fill_data` contains `symbol` or we have a way to get it.
            # For example, if the order that led to this fill had the symbol.

            # Placeholder: If symbol is not directly in fill_data, we need a mechanism to find it.
            # Perhaps from an internal cache of orders by ID.
            # For now, let's assume it's available or a default is used.
            # This part is CRITICAL and needs to be accurate.

            # Example: contract_id = fill_data.get('contractId')
            # symbol = self._get_symbol_for_contract_id(contract_id) # Needs implementation

            symbol_str = fill_data.get("symbol", "UNKNOWN_SYMBOL")  # Placeholder
            if (
                "contract" in fill_data and "name" in fill_data["contract"]
            ):  # Check if symbol is in contract details
                symbol_str = fill_data["contract"]["name"]

            fill_id = str(
                fill_data.get(
                    "tradeId", fill_data.get("id", f"unknown_fill_{time.time()}")
                )
            )
            order_id = str(fill_data["orderId"])
            qty = float(fill_data["qty"])
            price = float(fill_data["price"])
            # Tradovate 'action' is 'Buy' or 'Sell'
            side = "buy" if fill_data.get("action", "").lower() == "buy" else "sell"

            timestamp_str = fill_data.get(
                "timestamp"
            )  # e.g., "2023-04-01T10:30:00.123Z"
            if timestamp_str:
                # Ensure correct parsing of Tradovate's timestamp format
                dt_obj = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                dt_obj = datetime.now(timezone.utc)

            commission = float(fill_data.get("commission", 0.0))  # If available

            mapped_fill = Fill(
                id=fill_id,
                order_id=order_id,
                symbol=symbol_str,  # This needs to be correctly resolved
                qty=qty,
                price=price,
                side=side,
                timestamp=dt_obj,
                commission=commission,
            )
            logger.debug(f"Mapped Tradovate fill: {mapped_fill}")
            return mapped_fill
        except KeyError as e:
            logger.error(f"KeyError mapping Tradovate fill: {e}. Data: {fill_data}")
            return None
        except Exception as e:
            logger.error(
                f"Error mapping Tradovate fill: {e}. Data: {fill_data}", exc_info=True
            )
            return None

    def _map_tradovate_order_update(self, order_data: dict) -> Order | None:
        """Maps Tradovate order data to the internal Order model."""
        try:
            # This mapping is also speculative.
            # Required for Order: id, symbol, qty, side, order_type, status
            # Tradovate status might be: "Filled", "Working", "Canceled", "Rejected" etc.
            order_id = str(order_data["orderId"])

            # Symbol resolution needed here too.
            symbol_str = order_data.get("symbol", "UNKNOWN_SYMBOL")
            if "contract" in order_data and "name" in order_data["contract"]:
                symbol_str = order_data["contract"]["name"]

            qty = float(
                order_data.get("ordQty", 0)
            )  # 'ordQty' is often total order quantity

            action = order_data.get("action", "").lower()
            side = "buy" if action == "buy" else "sell"

            # Tradovate order types: "Market", "Limit", "Stop", etc.
            order_type_str = order_data.get("orderType", "unknown").lower()

            # Tradovate status mapping
            # Example: "Filled", "Working", "Canceled", "Rejected", "PartiallyFilled"
            tv_status = order_data.get("orderStatus", "unknown")
            status_map = {
                "filled": "filled",
                "working": "submitted",  # Or "new" or "open" depending on internal states
                "pendingfill": "partially_filled",  # Check actual Tradovate status for partial fills
                "canceled": "canceled",
                "rejected": "rejected",
                "partiallyfilled": "partially_filled",
                # Add more mappings as needed
            }
            internal_status = status_map.get(tv_status.lower(), "unknown")

            avg_fill_price = (
                float(order_data.get("avgPx", 0.0)) if order_data.get("avgPx") else None
            )
            filled_qty = float(
                order_data.get("cumQty", 0.0)
            )  # 'cumQty' is cumulative filled quantity

            limit_price_val = order_data.get("price")  # For limit orders
            limit_price = (
                float(limit_price_val) if limit_price_val is not None else None
            )

            stop_price_val = order_data.get("stopPrice")  # For stop orders
            stop_price = float(stop_price_val) if stop_price_val is not None else None

            # Timestamps: created, updated
            created_at_str = order_data.get(
                "timestamp"
            )  # Or specific created timestamp if available
            updated_at_str = order_data.get(
                "updatedTimestamp", created_at_str
            )  # Check field names

            created_at = (
                datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                if created_at_str
                else datetime.now(timezone.utc)
            )
            updated_at = (
                datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))
                if updated_at_str
                else created_at
            )

            mapped_order = Order(
                id=order_id,
                symbol=symbol_str,
                qty=qty,
                side=side,
                order_type=order_type_str,
                status=internal_status,
                filled_qty=filled_qty,
                avg_fill_price=avg_fill_price,
                limit_price=limit_price,
                stop_price=stop_price,
                created_at=created_at,
                updated_at=updated_at,
            )
            logger.debug(f"Mapped Tradovate order update: {mapped_order}")
            return mapped_order
        except KeyError as e:
            logger.error(f"KeyError mapping Tradovate order: {e}. Data: {order_data}")
            return None
        except Exception as e:
            logger.error(
                f"Error mapping Tradovate order: {e}. Data: {order_data}", exc_info=True
            )
            return None

    async def _get_initial_account_data(self):
        """Fetches initial account list to identify accountId if not set."""
        if self._account_id:  # Already have it
            return

        try:
            logger.info("Fetching account list to identify account ID...")
            response = await self._make_request("GET", "/account/list")
            accounts = response.json()
            if accounts and isinstance(accounts, list) and len(accounts) > 0:
                # Assuming we use the first account. Logic might be needed for multiple accounts.
                self._account_id = accounts[0].get("id")
                self._cash = float(
                    accounts[0].get("balance", 0.0)
                )  # Initial cash balance
                logger.info(
                    f"Account ID set to {self._account_id}, initial cash: {self._cash}"
                )

                # Store other relevant details from account object if needed
                # For example, user ID if it's different from token's userId
                # self.user_id_from_account = accounts[0].get("userId")

            else:
                logger.warning(
                    "No accounts found or empty response from /account/list."
                )
        except Exception as e:
            logger.error(f"Failed to fetch account list: {e}", exc_info=True)
            # Not raising here, as polling might eventually get it or connection can proceed without it for some ops

    async def _poll_data(self):
        """Periodically polls for cash and positions."""
        await asyncio.sleep(5)  # Initial delay to allow connection to establish fully

        while not self._stop_event.is_set():
            try:
                if (
                    not self.access_token_details
                    or self.access_token_details.is_expired
                ):
                    logger.info(
                        "Token expired or missing, skipping poll data iteration. Connect will handle re-auth."
                    )
                    await asyncio.sleep(POLL_INTERVAL)  # Wait before checking again
                    continue

                if not self._account_id:
                    await self._get_initial_account_data()
                    if not self._account_id:
                        logger.warning(
                            "Account ID not available, cannot poll for cash/positions accurately yet."
                        )
                        await asyncio.sleep(
                            POLL_INTERVAL
                        )  # Wait before retrying to get account_id
                        continue

                logger.debug("Polling for cash and positions...")
                # Fetch cash (though it might also come from WS props)
                try:
                    cash_data = (
                        await self.get_cash()
                    )  # This should use _make_request internally
                    # self._cash is updated by get_cash()
                    logger.debug(f"Polled cash: {self._cash}")
                except Exception as e_cash:
                    logger.error(f"Error polling cash: {e_cash}", exc_info=True)

                # Fetch positions
                try:
                    positions_data = (
                        await self.get_positions()
                    )  # This should use _make_request
                    # self._positions is updated by get_positions()
                    logger.debug(f"Polled positions: {len(self._positions)} items")
                except Exception as e_pos:
                    logger.error(f"Error polling positions: {e_pos}", exc_info=True)

                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                logger.info("Data poller task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in data poller: {e}", exc_info=True)
                # Avoid rapid spin on persistent error, wait before next iteration
                await asyncio.sleep(POLL_INTERVAL)
        logger.info("Data poller stopped.")

    # --- Implementation of BrokerAdapterBase abstract methods ---

    # connect() is already implemented above

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=0.5, max=2))
    async def submit_order(
        self,
        symbol: str,  # e.g., MESM25
        qty: float,
        side: str,  # 'buy' or 'sell'
        order_type: str,  # 'market', 'limit', 'stop'
        limit_price: float | None = None,
        stop_price: float | None = None,
        # Tradovate specific params might be needed
        # e.g., account_id, contract_id (needs mapping from symbol)
    ) -> Order | None:
        if not self._account_id:
            await self._get_initial_account_data()
            if not self._account_id:
                logger.error("Cannot submit order: Account ID not available.")
                # Consider raising an error or returning a rejected Order object
                return None  # Or an Order object with 'rejected' status

        # Map symbol to Tradovate contract ID - this is crucial
        # contract_id = self._resolve_symbol_to_contract_id(symbol)
        # For now, assume symbol is the Tradovate instrument name (e.g. "MESM25")
        # And that the API accepts it directly or implies contractId from it.

        order_payload = {
            "accountId": self._account_id,
            "action": "Buy" if side.lower() == "buy" else "Sell",
            "symbol": symbol,  # Tradovate's API expects the symbol string here
            "orderQty": int(
                qty
            ),  # Tradovate usually expects integer quantity for futures
            "orderType": order_type.capitalize(),  # e.g., "Market", "Limit"
            "isAutomated": True,  # As per prompt
            # "price": limit_price, # Only for Limit/StopLimit
            # "stopPrice": stop_price, # Only for Stop/StopLimit
        }

        if order_type.lower() == "limit":
            if limit_price is None:
                raise ValueError("limit_price must be provided for limit orders")
            order_payload["price"] = limit_price
        elif order_type.lower() == "stop":  # Assuming this means StopMarket
            if stop_price is None:
                raise ValueError("stop_price must be provided for stop orders")
            order_payload["orderType"] = (
                "Stop"  # Tradovate may differentiate StopLimit and StopMarket
            )
            order_payload["stopPrice"] = stop_price
        # Add other order types like StopLimit if needed

        logger.info(f"Submitting order: {order_payload}")
        try:
            response = await self._make_request(
                "POST", "/order/placeorder", json=order_payload
            )
            order_response_data = response.json()
            # Tradovate's placeorder response contains the order details including an orderId.
            # It might not be the full final state but an acknowledgement.
            # The actual fill/status updates will come via WebSocket.

            # We need to map this response to our internal Order model.
            # The response might look like: { "orderId": 12345, ... other fields ... }
            # It's an acknowledgement, status might be "Pending" or "Working"

            # For now, create a basic Order object. It will be updated by WebSocket.
            # This mapping needs to be accurate based on Tradovate's response.
            created_order = Order(
                id=str(order_response_data["orderId"]),
                symbol=symbol,
                qty=qty,
                side=side,
                order_type=order_type,
                status="submitted",  # Or map from response if available
                limit_price=limit_price,
                stop_price=stop_price,
                # filled_qty, avg_fill_price will be updated by fills via WS
            )
            logger.info(f"Order submitted successfully: {created_order.id}")
            return created_order
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Failed to submit order: {e.response.status_code} - {e.response.text}"
            )
            # Parse error for details if possible, e.g. margin error
            error_details = (
                e.response.json()
                if "application/json" in e.response.headers.get("content-type", "")
                else {"message": e.response.text}
            )
            # You might want to return an Order object with status 'rejected' and a reason
            # For now, returning None or re-raising
            raise  # Let retry handle it or propagate
        except Exception as e:
            logger.error(f"Unexpected error submitting order: {e}", exc_info=True)
            raise

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=0.5, max=2))
    async def cancel_order(self, order_id: str) -> bool:
        # order_id is expected to be a string representation of the integer ID.
        # Tradovate API docs should be consulted if the path parameter must be int.
        # For RESTful APIs, path parameters are typically strings that get converted server-side if needed.
        logger.info(f"Cancelling order: {order_id}")
        try:
            # Ensure order_id is an integer for the URL if Tradovate strictly requires it,
            # otherwise, string is fine. Assuming string for typical REST path construction.
            # If Tradovate expects an int in the path, this might need adjustment
            # or the server handles string-to-int conversion for path params.
            # For this example, we'll assume the string order_id is acceptable in the path.
            endpoint = (
                f"/order/cancelorder/{order_id}"  # Using order_id directly in path
            )

            # The prompt specifies DELETE /order/cancelorder/{orderId}.
            # No payload is typically sent with DELETE for this pattern.
            response = await self._make_request("DELETE", endpoint)

            logger.info(
                f"Order cancellation request for {order_id} successful. Status: {response.status_code}"
            )
            # Actual "canceled" status update should come via WebSocket.
            return True
        except httpx.HTTPStatusError as e:
            # Check if the error indicates the order was already cancelled or filled
            # Tradovate might return specific codes or messages for such cases.
            # For example, a 404 might mean order not found (could be already cancelled/filled and gone)
            # Or a 400/409 with a specific message body.
            logger.error(
                f"Failed to cancel order {order_id}: {e.response.status_code} - {e.response.text}"
            )
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error cancelling order {order_id}: {e}", exc_info=True
            )
            return False

    async def get_cash(self) -> dict[str, float]:
        if not self._account_id:  # If _account_id is not set, try to get it
            logger.info(
                "Account ID not set, attempting to fetch initial account data before getting cash."
            )
            await self._get_initial_account_data()
            if not self._account_id:
                logger.error("Failed to get account ID, cannot retrieve cash.")
                return (
                    self._cash
                )  # Returns {} if _get_initial_account_data didn't set it and it was initialized to {}

        try:
            response = await self._make_request("GET", "/account/list")
            if (
                response is None or response.json() is None
            ):  # Check if response or its json is None
                logger.warning(
                    "Failed to get account list or empty response from /account/list."
                )
                return (
                    self._cash
                )  # Return current self._cash (e.g., {} or last known good)

            accounts_data = response.json()
            if not isinstance(accounts_data, list):
                logger.error(
                    f"Expected list from /account/list, got {type(accounts_data)}"
                )
                return self._cash

            for acc_data in accounts_data:
                if acc_data.get("id") == self._account_id:
                    current_balance = float(acc_data.get("balance", 0.0))
                    buying_power = float(acc_data.get("buyingPower", 0.0))
                    currency_id = acc_data.get("currencyId")  # Assuming 1 is USD

                    # Margin Guard (example: /MES requires ~$50 margin per contract typically)
                    # This is a placeholder. Actual margin requirements vary by symbol and broker.
                    if (
                        buying_power < 50.0 and currency_id == 1
                    ):  # Assuming currencyId 1 is USD
                        logger.warning(
                            f"Margin Alert: Account {self._account_id} buying power ({buying_power} USD) is below $50."
                        )

                    # For now, assuming USD is the primary/only currency of interest
                    self._cash = {
                        "USD": current_balance
                    }  # Ensure self._cash is the dictionary

                    if self.runner:
                        asyncio.create_task(
                            self.runner.on_broker_event(
                                {"type": "cash_update", "data": self._cash}
                            )
                        )
                    return self._cash  # Explicitly return the dictionary self._cash

            logger.warning(
                f"Account {self._account_id} not found in /account/list response."
            )
            return (
                self._cash
            )  # Return current self._cash (e.g. {} or last known if not found)
        except Exception as e:
            logger.error(f"Failed to get cash balance: {e}", exc_info=True)
            return self._cash  # Return current self._cash on error

    async def get_positions(self) -> list[Position]:
        if not self._account_id:
            await self._get_initial_account_data()

        if not self._account_id:
            logger.warning("Cannot get positions: Account ID not available.")
            return self._positions  # Return last known or default

        logger.debug(f"Getting positions for account {self._account_id}...")
        try:
            # Tradovate endpoint: /position/list or /position/deps (dependencies, all positions for user)
            # Or /account/{id}/positions
            response = await self._make_request(
                "GET", "/position/list"
            )  # Or /position/deps
            tradovate_positions = response.json()

            current_positions = []
            if isinstance(tradovate_positions, list):
                for pos_data in tradovate_positions:
                    # Filter by accountId if /position/list returns for multiple accounts
                    if pos_data.get("accountId") == self._account_id:
                        # Map Tradovate position data to our internal Position model
                        # Required: symbol, qty, avg_entry_price
                        # Tradovate data might have 'contractId', 'qty', 'avgEntryPrice', 'short', 'long'

                        symbol_str = "UNKNOWN_SYMBOL"
                        if "contract" in pos_data and "name" in pos_data["contract"]:
                            symbol_str = pos_data["contract"]["name"]
                        elif (
                            "name" in pos_data
                        ):  # If symbol name is directly in position item
                            symbol_str = pos_data["name"]

                        qty = float(
                            pos_data.get("netPos", 0.0)
                        )  # 'netPos' often used for net quantity
                        if qty == 0:
                            continue  # Skip zero positions

                        avg_price = float(pos_data.get("avgEntryPrice", 0.0))

                        # If Tradovate provides 'longQty' and 'shortQty' separately:
                        # long_qty = float(pos_data.get("longQty", 0))
                        # short_qty = float(pos_data.get("shortQty", 0))
                        # if long_qty > 0:
                        #     qty = long_qty
                        # elif short_qty > 0:
                        #     qty = -short_qty # Negative for short
                        # else:
                        #     continue # No position

                        current_positions.append(
                            Position(
                                symbol=symbol_str,  # Needs correct mapping/resolution
                                qty=qty,
                                avg_entry_price=avg_price,
                            )
                        )
            self._positions = current_positions
            logger.info(
                f"Positions updated: {len(self._positions)} open positions for account {self._account_id}"
            )
            return self._positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}", exc_info=True)
            return self._positions  # Return last known positions on error

    async def close(self):
        logger.info("Closing TradovateBrokerAdapter...")
        if self.is_disconnecting:  # Already in process of closing
            logger.info("Close already in progress.")
            return

        self.is_disconnecting = True  # Signal intent to disconnect
        self._stop_event.set()  # Signal all background tasks to stop

        tasks_to_await = []

        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            tasks_to_await.append(self.heartbeat_task)

        if self.ws_listener_task:
            self.ws_listener_task.cancel()
            tasks_to_await.append(self.ws_listener_task)

        if self._data_poller_task:
            self._data_poller_task.cancel()
            tasks_to_await.append(self._data_poller_task)

        # Wait for tasks to complete cancellation
        for task in tasks_to_await:
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Task {task.get_name()} cancelled successfully.")
            except Exception as e:
                logger.error(
                    f"Error during cancellation of task {task.get_name()}: {e}"
                )

        if self.ws_connection and not self.ws_connection.closed:
            logger.info("Closing WebSocket connection.")
            try:
                await self.ws_connection.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket connection: {e}")
        self.ws_connection = None  # Clear the connection object

        if self.http_client:
            logger.info("Closing HTTP client.")
            await self.http_client.aclose()
            # self.http_client = None # httpx.AsyncClient doesn't need to be None'd, aclose() is enough

        self.access_token_details = None  # Clear token
        self._account_id = None  # Clear account context
        self.is_connecting = False  # Reset connecting flag
        # self.is_disconnecting should remain True or be reset if class can be reused.
        # For now, assume close means it's done.

        logger.info("TradovateBrokerAdapter closed.")
        if self.runner:  # Notify runner if available
            asyncio.create_task(
                self.runner.on_broker_event({"type": "disconnect", "status": "success"})
            )

    # --- Helper methods for symbol mapping (placeholders) ---
    # These would interact with Tradovate's contract/instrument APIs
    # async def _resolve_symbol_to_contract_id(self, symbol: str) -> int:
    #     # Placeholder: implement logic to find Tradovate contractId for a given symbol string
    #     # This might involve a lookup via /contract/suggest or /contract/item
    #     logger.debug(f"Resolving symbol '{symbol}' to contractId (not implemented, using placeholder).")
    #     # Example: if symbol is "MESM25", contractId might be 123456
    #     # This is a critical piece for actual trading.
    #     # For now, assuming the 'symbol' field in order placement takes the string name.
    #     if symbol == "MESM25": return 783338 # Example placeholder ID, NOT REAL
    #     raise ValueError(f"Symbol {symbol} not resolvable to a Tradovate contractId (dummy implementation).")

    # async def _get_symbol_for_contract_id(self, contract_id: int) -> str:
    #     # Placeholder: implement logic to find symbol string for a given Tradovate contractId
    #     logger.debug(f"Fetching symbol for contractId {contract_id} (not implemented).")
    #     # Example: /contract/item/{contract_id}
    #     # For now, returning a placeholder
    #     return f"CONTRACT_{contract_id}"

    #     # For now, returning a placeholder
    #     return f"CONTRACT_{contract_id}"
