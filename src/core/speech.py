import speech_recognition as sr
from google.cloud import speech
from src.utils.logger import logger
from src.utils.errors import SpeechRecognitionError
from src.config import Config

class SpeechRecognizer:
    def __init__(self):
        """Initialize the speech recognizer."""
        try:
            self.recognizer = sr.Recognizer()
            self.client = speech.SpeechClient()
            logger.info("SpeechRecognizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SpeechRecognizer: {str(e)}")
            raise SpeechRecognitionError(f"Speech recognition initialization failed: {str(e)}")

    def recognize_from_microphone(self):
        """Recognize speech from microphone input."""
        try:
            with sr.Microphone() as source:
                logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source)
                logger.info("Listening for speech input...")
                audio = self.recognizer.listen(source)
                
                # Convert to Google Cloud Speech format
                audio_data = audio.get_wav_data()
                audio = speech.RecognitionAudio(content=audio_data)
                
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=Config.SPEECH_SAMPLE_RATE,
                    language_code=Config.SPEECH_LANGUAGE,
                    enable_automatic_retranscription=True
                )
                
                response = self.client.recognize(config=config, audio=audio)
                
                if not response.results:
                    raise SpeechRecognitionError("No speech detected")
                
                transcript = response.results[0].alternatives[0].transcript
                logger.info(f"Successfully recognized speech: {transcript[:50]}...")
                return transcript
        except Exception as e:
            logger.error(f"Error in recognize_from_microphone method: {str(e)}")
            raise SpeechRecognitionError(f"Speech recognition failed: {str(e)}")

    def recognize_from_file(self, audio_file_path):
        """Recognize speech from an audio file."""
        try:
            with open(audio_file_path, "rb") as audio_file:
                content = audio_file.read()
            
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=Config.SPEECH_SAMPLE_RATE,
                language_code=Config.SPEECH_LANGUAGE
            )
            
            response = self.client.recognize(config=config, audio=audio)
            
            if not response.results:
                raise SpeechRecognitionError("No speech detected in file")
            
            transcript = response.results[0].alternatives[0].transcript
            logger.info(f"Successfully recognized speech from file: {transcript[:50]}...")
            return transcript
        except Exception as e:
            logger.error(f"Error in recognize_from_file method: {str(e)}")
            raise SpeechRecognitionError(f"File speech recognition failed: {str(e)}")

# Create a global instance
speech_recognizer = SpeechRecognizer()

def recognize_speech():
    """Global function to recognize speech using the speech recognizer."""
    return speech_recognizer.recognize_from_microphone() 