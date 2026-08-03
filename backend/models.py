from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints


FeedbackType = Literal[
    "relevant", "inspiring", "not_useful", "save_for_later", "already_known", "read"
]
RankingMode = Literal["strict", "balanced", "exploratory"]
SourcePreference = Literal["boost", "normal", "lower", "exclude"]

ProfilePhrase = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=600),
]


class ResearchProfile(BaseModel):
    name: str = Field(default="Researcher", min_length=1, max_length=120)
    headline: str = Field(
        default="Research interests not configured", min_length=1, max_length=600
    )
    domains: list[ProfilePhrase] = Field(default_factory=list, max_length=50)
    methods: list[ProfilePhrase] = Field(default_factory=list, max_length=50)
    systems: list[ProfilePhrase] = Field(default_factory=list, max_length=50)
    current_questions: list[ProfilePhrase] = Field(default_factory=list, max_length=50)
    adjacent_fields: list[ProfilePhrase] = Field(default_factory=list, max_length=50)
    keywords: list[ProfilePhrase] = Field(default_factory=list, max_length=100)


class SettingsUpdate(BaseModel):
    top_n: int | None = Field(default=None, ge=1, le=100)
    first_sync_days: int | None = Field(default=None, ge=1, le=30)
    candidate_multiplier: int | None = Field(default=None, ge=1, le=5)
    ranking_mode: RankingMode | None = None
    source_preferences: dict[str, SourcePreference] | None = None
    folder_preferences: dict[str, SourcePreference] | None = None


class FeedbackRequest(BaseModel):
    value: FeedbackType


class ProfileUpdate(BaseModel):
    profile: ResearchProfile


class RecommendationResult(BaseModel):
    article_id: str
    relevance_score: float = Field(ge=0, le=1)
    novelty_score: float = Field(ge=0, le=1)
    inspiration_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    reason: str
    core_finding: str
    innovation: str
    connection: str
    idea: str
    idea_is_speculative: bool = True
    labels: list[str] = Field(default_factory=list)
