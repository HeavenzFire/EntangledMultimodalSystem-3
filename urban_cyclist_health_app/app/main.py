import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import json
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Urban Cyclist Health App",
    description="A comprehensive health and wellness app for urban cyclists with cancer/mental health support",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class User(BaseModel):
    id: str
    name: str
    email: str
    health_conditions: List[str] = []
    preferences: dict = {}

class Ride(BaseModel):
    id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    distance: float
    route: List[dict]
    mood_rating: int
    stress_level: int
    notes: Optional[str] = None

class MentalHealthCheck(BaseModel):
    id: str
    user_id: str
    timestamp: datetime
    phq9_score: int
    anxiety_level: int
    voice_analysis: Optional[dict] = None
    notes: Optional[str] = None

# In-memory storage (replace with database in production)
users = {}
rides = {}
mental_health_checks = {}

@app.post("/users/")
async def create_user(user: User):
    if user.id in users:
        raise HTTPException(status_code=400, detail="User already exists")
    users[user.id] = user
    return user

@app.get("/users/{user_id}")
async def get_user(user_id: str):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    return users[user_id]

@app.post("/rides/")
async def create_ride(ride: Ride):
    if ride.user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    rides[ride.id] = ride
    return ride

@app.get("/rides/{user_id}")
async def get_user_rides(user_id: str):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    return [ride for ride in rides.values() if ride.user_id == user_id]

@app.post("/mental-health/")
async def create_mental_health_check(check: MentalHealthCheck):
    if check.user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    mental_health_checks[check.id] = check
    return check

@app.get("/mental-health/{user_id}")
async def get_mental_health_history(user_id: str):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    return [check for check in mental_health_checks.values() if check.user_id == user_id]

@app.get("/health-summary/{user_id}")
async def get_health_summary(user_id: str):
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_rides = [ride for ride in rides.values() if ride.user_id == user_id]
    user_checks = [check for check in mental_health_checks.values() if check.user_id == user_id]
    
    return {
        "total_rides": len(user_rides),
        "total_distance": sum(ride.distance for ride in user_rides),
        "average_mood": sum(ride.mood_rating for ride in user_rides) / len(user_rides) if user_rides else 0,
        "average_stress": sum(ride.stress_level for ride in user_rides) / len(user_rides) if user_rides else 0,
        "latest_phq9_score": user_checks[-1].phq9_score if user_checks else None,
        "mental_health_trend": "improving" if len(user_checks) > 1 and user_checks[-1].phq9_score < user_checks[0].phq9_score else "stable"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 