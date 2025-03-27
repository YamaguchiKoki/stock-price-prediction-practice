from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

class EnvConfig(BaseSettings):
    input_dir: str = Field(default='data')

    class Config:
        env_file = '.env'
        extra = 'ignore'

    @property
    def root_dir(self) -> Path:
        return Path(__file__).parent.parent.parent

    @property
    def abs_input_dir(self) -> Path:
        return (self.root_dir / self.input_dir).resolve()
