from typing import List, Optional
from sqlmodel import Field, Relationship, SQLModel, UniqueConstraint
from pydantic import constr
from datetime import datetime

class NASBench201ModelBase(SQLModel):
    name: str = Field(index=True)
    framework: str = Field(index=True)
    dataset_version: float = Field(index=True)
    model_type: constr(regex="^(cells|ops)$") = Field(index=True)
    device: str = Field(index=True, default="any")
    input_dim: int = Field(index=True)
    arch_idx: int = Field(index=True, default=-1)
    
    # Number of bytes taken up by the model on disk
    size_bytes: int

class NASBench201Model(NASBench201ModelBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    path: str

    creation_date: datetime = Field(default_factory=datetime.now)
    profiling_results: List["ProfilingResult"] = Relationship(
        back_populates="model")
    
    # Add a unique constraint for the combination of name, framework, dataset_version, and device
    __table_args__ = (UniqueConstraint("name", "framework", "dataset_version", "device", "input_dim"),)

# class NASBench201ModelCreate(NASBench201ModelBase):
#     pass

# class NASBench201ModelRead(NASBench201ModelBase):
#     file: FileResponse

class ProfilingResult(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    inf_latency_mean: float
    inf_latency_std: float
    num_runs: int
    inf_latencies: List[float]
    performance_counters: List["PerfCounter"] = Relationship(
        back_populates="profiling_result")

    experiment_date: datetime = Field(default_factory=datetime.now)
    processor: constr(regex="^(cpu|gpu)$") = Field(index=True)

    model_id: int = Field(foreign_key="nasbench201model.id")
    model: NASBench201Model = Relationship(back_populates="profiling_results")

class PerfCounter(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    value: float

    profiling_result_id: int = Field(
        foreign_key="profilingresult.id")
    profiling_result: ProfilingResult = Relationship(
        back_populates="performance_counters")
