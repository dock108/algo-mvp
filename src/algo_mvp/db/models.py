from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    broker_order_id = Column(String, nullable=False, index=True)
    symbol = Column(String, nullable=False, index=True)
    side = Column(String, nullable=False)
    order_type = Column(String, nullable=False)
    qty = Column(Float, nullable=False)
    limit_price = Column(Float, nullable=True)
    stop_price = Column(Float, nullable=True)
    status = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False)

    fills = relationship("Fill", back_populates="order")

    __table_args__ = (UniqueConstraint("broker_order_id", name="uq_broker_order_id"),)


class Fill(Base):
    __tablename__ = "fills"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    fill_qty = Column(Float, nullable=False)
    fill_price = Column(Float, nullable=False)
    commission = Column(Float, nullable=False, default=0.0)
    filled_at = Column(DateTime, nullable=False, index=True)

    order = relationship("Order", back_populates="fills")


class Equity(Base):
    __tablename__ = "equity"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    equity = Column(Float, nullable=False)


class Log(Base):
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    level = Column(String, nullable=False)
    message = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, index=True)
