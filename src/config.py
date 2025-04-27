from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    OPENAI_API_KEY: SecretStr

config = Config(_env_file=".env")   # pyright: ignore
