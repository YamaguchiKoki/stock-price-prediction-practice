from pydantic import Field
from pydantic_settings import BaseSettings

class EnvConfig(BaseSettings):
    input_dir: str = Field(default='../data/input')

    class Config:
        env_file = '.env'
        extra = 'ignore'
