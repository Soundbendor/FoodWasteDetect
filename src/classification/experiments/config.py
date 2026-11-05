"""Mutable configuration validated at initialization for whole-image experiments."""

from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PositiveInt = Annotated[int, Field(gt=0)]
NonemptyString = Annotated[str, Field(min_length=1)]
PositiveSeconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class ConfigModel(BaseModel):
    """Reject unknown fields and coercion at initialization, including in defaults."""

    model_config = ConfigDict(strict=True, extra="forbid", validate_default=True)


class DatasetConfig(ConfigModel):
    """Local FoodX251 annotation location."""

    name: NonemptyString
    root: NonemptyString

    @field_validator("name")
    @classmethod
    def supported_dataset(cls, value: str) -> str:
        if value.lower() != "foodx251":
            raise ValueError("Only foodx251 is supported")
        return value


class VertexConfig(ConfigModel):
    """Cloud settings; project is required only when constructing the backend."""

    project: NonemptyString | None = None
    location: NonemptyString = "global"
    max_concurrency: PositiveInt = 4
    max_attempts: PositiveInt = 5
    request_timeout: PositiveSeconds = 60.0
    classification_prompt: bool = False


class EmbeddingConfig(ConfigModel):
    """Encoder selection and bounded input batches."""

    model_id: NonemptyString
    batch_size: PositiveInt = 32
    device: NonemptyString = "auto"
    truncate_dim: PositiveInt | None = None
    vertex: VertexConfig = Field(default_factory=VertexConfig)


class LanceDBConfig(ConfigModel):
    """Reference and optional validation table destinations."""

    uri: NonemptyString
    table: NonemptyString
    validation_table: str | None = None


class OllamaConfig(ConfigModel):
    """Owned server connection and process settings, also used for downloads."""

    binary_path: NonemptyString
    host: NonemptyString
    startup_timeout: PositiveSeconds
    request_timeout: PositiveSeconds
    llm: NonemptyString | None = None

    @field_validator("binary_path", "host", "llm")
    @classmethod
    def nonblank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must be a nonempty string")
        return value


class ExperimentConfig(ConfigModel):
    """Validated settings passed through every experiment stage."""

    dataset: DatasetConfig
    embedding: EmbeddingConfig
    lancedb: LanceDBConfig
    output_dir: NonemptyString
    k: PositiveInt
    embed_images: bool
    embed_class_labels: bool
    embed_descriptions: bool
    vector_dimension: PositiveInt | None = None
    sweep_dimensions: list[PositiveInt] | None = None
    n_imgs_per_class: PositiveInt | None = None
    descriptions_file: str | None = None
    cache_validation_embeddings: bool = False
    ollama: OllamaConfig | None = None
    description_count: PositiveInt | None = None

    @field_validator("sweep_dimensions")
    @classmethod
    def unique_sweep_dimensions(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and (not value or len(set(value)) != len(value)):
            raise ValueError("sweep_dimensions must be nonempty and unique")
        return value

    @model_validator(mode="after")
    def experiment_constraints(self) -> Self:
        if not any((self.embed_images, self.embed_class_labels, self.embed_descriptions)):
            raise ValueError("Select at least one reference source")
        output = Path(self.output_dir).expanduser().resolve()
        if output.is_relative_to(Path(self.dataset.root).expanduser().resolve()):
            raise ValueError("Output directory must be separate from raw data")
        if self.cache_validation_embeddings:
            name = self.lancedb.validation_table
            if not name or not name.strip() or name == self.lancedb.table:
                raise ValueError(
                    "lancedb.validation_table must be nonempty and distinct from reference table"
                )
        return self
