from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum

class HealthCondition(str, Enum):
    CANCER = "cancer"
    DEPRESSION = "depression"
    ANXIETY = "anxiety"
    PTSD = "ptsd"
    OTHER = "other"

class UserPreferences(BaseModel):
    dark_mode: bool = True
    notifications_enabled: bool = True
    share_health_data: bool = False
    preferred_units: str = "metric"
    language: str = "en"

class User(BaseModel):
    id: str = Field(..., description="Unique identifier for the user")
    name: str = Field(..., description="User's full name")
    email: str = Field(..., description="User's email address")
    health_conditions: List[HealthCondition] = Field(default_factory=list)
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class RoutePoint(BaseModel):
    latitude: float
    longitude: float
    timestamp: datetime
    elevation: Optional[float] = None
    heart_rate: Optional[int] = None
    speed: Optional[float] = None

class RideMetrics(BaseModel):
    distance: float = Field(..., description="Total distance in kilometers")
    duration: float = Field(..., description="Duration in seconds")
    average_speed: float = Field(..., description="Average speed in km/h")
    max_speed: float = Field(..., description="Maximum speed in km/h")
    elevation_gain: float = Field(..., description="Total elevation gain in meters")
    calories_burned: float = Field(..., description="Estimated calories burned")

class Ride(BaseModel):
    id: str = Field(..., description="Unique identifier for the ride")
    user_id: str = Field(..., description="ID of the user who completed the ride")
    start_time: datetime = Field(..., description="When the ride started")
    end_time: Optional[datetime] = Field(None, description="When the ride ended")
    route: List[RoutePoint] = Field(..., description="GPS points along the route")
    metrics: RideMetrics = Field(..., description="Ride performance metrics")
    mood_rating: int = Field(..., ge=1, le=10, description="Mood rating from 1-10")
    stress_level: int = Field(..., ge=1, le=10, description="Stress level from 1-10")
    notes: Optional[str] = Field(None, description="Optional notes about the ride")
    weather_conditions: Optional[Dict[str, Any]] = Field(None, description="Weather data during the ride")

class PHQ9Question(BaseModel):
    question: str
    score: int = Field(..., ge=0, le=3)

class MentalHealthCheck(BaseModel):
    id: str = Field(..., description="Unique identifier for the check")
    user_id: str = Field(..., description="ID of the user")
    timestamp: datetime = Field(..., description="When the check was performed")
    phq9_questions: List[PHQ9Question] = Field(..., description="PHQ-9 questionnaire responses")
    phq9_score: int = Field(..., ge=0, le=27, description="Total PHQ-9 score")
    anxiety_level: int = Field(..., ge=1, le=10, description="Anxiety level from 1-10")
    voice_analysis: Optional[Dict[str, Any]] = Field(None, description="Results from voice analysis")
    notes: Optional[str] = Field(None, description="Optional notes about the check")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions based on the check")

class HealthSummary(BaseModel):
    user_id: str = Field(..., description="ID of the user")
    period_start: datetime = Field(..., description="Start of the summary period")
    period_end: datetime = Field(..., description="End of the summary period")
    total_rides: int = Field(..., description="Total number of rides in the period")
    total_distance: float = Field(..., description="Total distance covered in kilometers")
    average_mood: float = Field(..., description="Average mood rating")
    average_stress: float = Field(..., description="Average stress level")
    mental_health_trend: str = Field(..., description="Trend in mental health (improving/stable/declining)")
    latest_phq9_score: Optional[int] = Field(None, description="Most recent PHQ-9 score")
    recommended_activities: List[str] = Field(default_factory=list, description="Recommended activities based on the summary") 