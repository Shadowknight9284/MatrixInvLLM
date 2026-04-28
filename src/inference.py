import os
import logging
try:
    from tinker import ServiceClient
except ImportError:
    ServiceClient = None
    logger = logging.getLogger(__name__)
    logger.warning("tinker library not found. Inference will return mock data.")

logger = logging.getLogger(__name__)

def get_tinker_client():
    """
    Initialize and return the Tinker ServiceClient.
    Expects TINKER_API_KEY to be set in environment.
    """
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        logger.warning("TINKER_API_KEY not found in environment.")
    
    return ServiceClient(api_key=api_key)

def sample_model(prompt, config, model_id=None):
    """
    Sample from a Tinker model using the SamplingClient.
    """
    inf_cfg = config.get("inference", {})
    if model_id is None:
        model_id = inf_cfg.get("base_model", "meta-llama/Llama-2-7b") # default fallback
    
    if ServiceClient is None:
        # Mock Response for verification: correctly formatted but random numbers
        return "Inverse:\n0.1, 0.2\n0.3, 0.4"
        
    client = get_tinker_client()
    
    # Configure parameters
    params = {
        "max_new_tokens": inf_cfg.get("max_new_tokens", 256),
        "temperature": inf_cfg.get("temperature", 0.0),
        "stop_sequences": inf_cfg.get("stop_sequences", []),
    }
    
    try:
        # According to requirements: 'create or obtain the appropriate Tinker sampling client'
        sampling_client = client.sampling(model=model_id)
        
        response = sampling_client.sample(
            prompt=prompt,
            **params
        )
        
        return response.text
    except Exception as e:
        logger.error(f"Tinker inference error: {e}")
        return None
