import enum
import uuid
from datetime import date,datetime
from typing import Any

from sqlalchemy import String, Date, Integer, Text, Enum, DateTime, func, ForeignKey, Boolean, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

def new_id():
    return str(uuid.uuid4())
class DocumentStatus(str,enum.Enum):
    pending = "pending"
    indexing = "indexing"
    ready = "ready"
    failed = "failed"

class TimestampMixin:
    created_at:Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now())
    updated_at:Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

class PolicyDocument(TimestampMixin,Base):
    __tablename__ = "policy_documents"
    id:Mapped[str] = mapped_column(String(36),primary_key=True,default=new_id)
    # mapped_column 的第一个参数是数据库里真实的列名。
    # 库里已经是 file_name / effective_data，保留列名不动就能免去一次数据库迁移。
    filename:Mapped[str] = mapped_column("file_name",String(225))
    stored_path:Mapped[str] = mapped_column(String(500))
    content_type:Mapped[str] =mapped_column(String(100))
    topic:Mapped[str |None] = mapped_column(String(100),index=True)
    # 注意 effective_data 是早期的拼写错误，列名保持原样，Python 侧统一用 effective_date
    effective_date:Mapped[date | None] = mapped_column("effective_data",Date,index=True)
    status:Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus),default=DocumentStatus.pending,index=True
    )
    error_message:Mapped[str | None] = mapped_column(Text)
    chunk_count:Mapped[int] = mapped_column(Integer,default=0)
    index_version:Mapped[int]=mapped_column(Integer,default=0)

    chunks:Mapped[list["DocumentChunk"]]=relationship(
        back_populates="document",cascade="all,delete-orphan"
    )

class DocumentChunk(TimestampMixin,Base):
    __tablename__ = "document_chunks"

    id:Mapped[str] = mapped_column(String(36),primary_key=True,default=new_id)
    # index 是列级选项，不能塞进 ForeignKey 里（ForeignKey 只认 ondelete/onupdate，
    # 其余 kwargs 会被当成方言参数校验并直接报 TypeError）
    document_id:Mapped[str] = mapped_column(
        ForeignKey("policy_documents.id",ondelete="CASCADE"),index=True
    )
    parent_id:Mapped[str | None] = mapped_column(
        ForeignKey("document_chunks.id",ondelete="CASCADE"),index=True
    )
    ordinal:Mapped[int] = mapped_column(Integer)
    section:Mapped[str | None] = mapped_column(String(255))
    content:Mapped[str | None] = mapped_column(Text)
    is_parent:Mapped[bool] = mapped_column(Boolean,default=False,index=True)
    token_count:Mapped[int] = mapped_column(Integer,default=0)
    metadata_json:Mapped[dict[str,Any]] = mapped_column(JSON,default=dict)

    document:Mapped[PolicyDocument]=relationship(back_populates="chunks")
