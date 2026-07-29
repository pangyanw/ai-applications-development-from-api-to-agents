"""
Configuration constants for AI service integrations.

This module centralizes all API endpoints, API keys, and default configuration
values used across different AI service providers (OpenAI, Anthropic, Gemini).

All API keys are loaded from environment variables for security.
"""

import os

# Default system prompt used across all AI services
DEFAULT_SYSTEM_PROMPT = "You are an assistant who answers concisely and informatively."

# OpenAI API configuration
OPENAI_HOST = "https://api.openai.com"
OPENAI_CHAT_COMPLETIONS_ENDPOINT = f"{OPENAI_HOST}/v1/chat/completions"
OPENAI_RESPONSES_ENDPOINT = f"{OPENAI_HOST}/v1/responses"
OPENAI_EMBEDDINGS_ENDPOINT = f"{OPENAI_HOST}/v1/embeddings"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# Anthropic API configuration
# ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
# Zhipu Compatible
ANTHROPIC_ENDPOINT = "https://open.bigmodel.cn/api/anthropic"
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')

# Google Gemini API configuration
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

# Zhipu AI API configuration
ZHIPUAI_ENDPOINT = "https://open.bigmodel.cn/api/paas/v3/model-api"
ZHIPUAI_API_KEY = os.getenv('ZHIPUAI_API_KEY', '')
ZHIPUAI_OPENAI_API_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

# User Service API configuration
USER_SERVICE_ENDPOINT = "http://localhost:8041"