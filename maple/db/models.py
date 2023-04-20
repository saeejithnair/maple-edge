from typing import List, Optional
from enum import Enum
from sqlmodel import Field, Relationship, SQLModel
from pydantic import constr
from datetime import datetime


class NASBench201ModelBase(SQLModel):
    name: str
    framework: str
    dataset_version: float
    model_type: constr(regex="^(cells|ops)$")
    device: Optional[str] = None

class NASBench201Model(NASBench201ModelBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    path: str

    profiling_results: List["ProfilingResult"] = Relationship(
        back_populates="model")

class ProfilingResult(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    inf_latency_mean: float
    inf_latency_std: float
    num_runs: int
    inf_latencies: List[float]
    performance_counters: List["PerfCounter"] = Relationship(
        back_populates="profiling_result")

    experiment_date: datetime = Field(default_factory=datetime.now)
    processor: constr(regex="^(cpu|gpu)$")

    model_id: int = Field(foreign_key="nasbench201model.id")
    model: NASBench201Model = Relationship(back_populates="profiling_results")

class PerfCounter(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    value: float

    profiling_result_id: int = Field(
        foreign_key="profilingresult.id")
    profiling_result: ProfilingResult = Relationship(
        back_populates="performance_counters")
