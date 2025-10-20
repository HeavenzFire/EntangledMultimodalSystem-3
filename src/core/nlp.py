from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.utils.logger import logger
from src.utils.errors import ModelError
from src.config import Config

class NLPProcessor:
    def __init__(self):
        """Initialize the NLP processor with GPT-2 model."""
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
            logger.info("NLPProcessor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NLPProcessor: {str(e)}")
            raise ModelError(f"NLP initialization failed: {str(e)}")

    def generate_text(self, prompt, max_length=150, num_return_sequences=1):
        """Generate text based on the given prompt."""
        try:
            if not prompt or not isinstance(prompt, str):
                raise ValueError("Prompt must be a non-empty string")
            
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_k=50,
                top_p=0.95
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Successfully generated text for prompt: {prompt[:50]}...")
            return generated_text
        except Exception as e:
            logger.error(f"Error in generate_text method: {str(e)}")
            raise ModelError(f"Text generation failed: {str(e)}")

    def analyze_sentiment(self, text):
        """Analyze the sentiment of the given text."""
        try:
            if not text or not isinstance(text, str):
                raise ValueError("Text must be a non-empty string")
            
            # Simple sentiment analysis implementation
            positive_words = {"good", "great", "excellent", "wonderful", "amazing"}
            negative_words = {"bad", "terrible", "awful", "horrible", "poor"}
            
            words = text.lower().split()
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            sentiment = "positive" if positive_count > negative_count else "negative" if negative_count > positive_count else "neutral"
            logger.info(f"Successfully analyzed sentiment for text: {text[:50]}...")
            return {
                "sentiment": sentiment,
                "positive_words": positive_count,
                "negative_words": negative_count
            }
        except Exception as e:
            logger.error(f"Error in analyze_sentiment method: {str(e)}")
            raise ModelError(f"Sentiment analysis failed: {str(e)}")

# Create a global instance
nlp_processor = NLPProcessor()

def generate_text(prompt):
    """Global function to generate text using the NLP processor."""
    return nlp_processor.generate_text(prompt) 