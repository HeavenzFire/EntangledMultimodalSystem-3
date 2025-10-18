import os
import requests
from typing import Optional, Dict, Any
import json
from datetime import datetime

class StravaAPI:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://www.strava.com/api/v3"
        self.access_token = None
        
    def authenticate(self, code: str) -> bool:
        """Authenticate with Strava API"""
        try:
            response = requests.post(
                "https://www.strava.com/oauth/token",
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": code,
                    "grant_type": "authorization_code"
                }
            )
            response.raise_for_status()
            self.access_token = response.json()["access_token"]
            return True
        except Exception as e:
            print(f"Strava authentication failed: {str(e)}")
            return False
            
    def get_activities(self, start_date: datetime, end_date: datetime) -> list:
        """Get user activities from Strava"""
        if not self.access_token:
            raise ValueError("Not authenticated with Strava")
            
        try:
            response = requests.get(
                f"{self.base_url}/athlete/activities",
                headers={"Authorization": f"Bearer {self.access_token}"},
                params={
                    "after": int(start_date.timestamp()),
                    "before": int(end_date.timestamp())
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Failed to get Strava activities: {str(e)}")
            return []

class KintsugiAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.kintsugihealth.com/v1"
        
    def analyze_voice(self, audio_file_path: str) -> Optional[Dict[str, Any]]:
        """Analyze voice for depression detection"""
        try:
            with open(audio_file_path, "rb") as audio_file:
                response = requests.post(
                    f"{self.base_url}/voice/analyze",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={"audio": audio_file}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"Voice analysis failed: {str(e)}")
            return None
            
    def get_depression_score(self, analysis_result: Dict[str, Any]) -> float:
        """Extract depression score from voice analysis"""
        return analysis_result.get("depression_score", 0.0)

class HealthKitAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.healthkit.com/v1"
        
    def get_health_metrics(self, user_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get health metrics from HealthKit"""
        try:
            response = requests.get(
                f"{self.base_url}/users/{user_id}/metrics",
                headers={"Authorization": f"Bearer {self.api_key}"},
                params={
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Failed to get HealthKit metrics: {str(e)}")
            return {}
            
    def get_mental_health_data(self, user_id: str) -> Dict[str, Any]:
        """Get mental health data from HealthKit"""
        try:
            response = requests.get(
                f"{self.base_url}/users/{user_id}/mental_health",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Failed to get mental health data: {str(e)}")
            return {}

class WithingsAPI:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://wbsapi.withings.net/v2"
        self.access_token = None
        
    def authenticate(self, code: str) -> bool:
        """Authenticate with Withings API"""
        try:
            response = requests.post(
                "https://account.withings.com/oauth2/token",
                data={
                    "grant_type": "authorization_code",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": code,
                    "redirect_uri": "https://your-app.com/callback"
                }
            )
            response.raise_for_status()
            self.access_token = response.json()["access_token"]
            return True
        except Exception as e:
            print(f"Withings authentication failed: {str(e)}")
            return False
            
    def get_heart_rate(self, start_date: datetime, end_date: datetime) -> list:
        """Get heart rate data from Withings"""
        if not self.access_token:
            raise ValueError("Not authenticated with Withings")
            
        try:
            response = requests.get(
                f"{self.base_url}/measure",
                headers={"Authorization": f"Bearer {self.access_token}"},
                params={
                    "action": "getmeas",
                    "meastype": 11,  # Heart rate
                    "startdate": int(start_date.timestamp()),
                    "enddate": int(end_date.timestamp())
                }
            )
            response.raise_for_status()
            return response.json()["body"]["measuregrps"]
        except Exception as e:
            print(f"Failed to get Withings heart rate data: {str(e)}")
            return []
            
    def get_stress_level(self) -> Optional[float]:
        """Get current stress level from Withings"""
        if not self.access_token:
            raise ValueError("Not authenticated with Withings")
            
        try:
            response = requests.get(
                f"{self.base_url}/measure",
                headers={"Authorization": f"Bearer {self.access_token}"},
                params={
                    "action": "getmeas",
                    "meastype": 88  # Stress level
                }
            )
            response.raise_for_status()
            return response.json()["body"]["measuregrps"][0]["measures"][0]["value"]
        except Exception as e:
            print(f"Failed to get Withings stress level: {str(e)}")
            return None 