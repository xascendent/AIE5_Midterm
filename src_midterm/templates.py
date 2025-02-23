from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import List, Optional
from logger import logger

class MetaDataModel(BaseModel):
    """Standard metadata model with required and optional fields."""
    
    # REQUIRED fields
    document_id: str = Field(..., description="Unique document identifier")
    document_date: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        description="Date of the document creation"
    )
    title: str = Field(..., min_length=1, max_length=255, description="Title of the document")

    # OPTIONAL fields
    description: Optional[str] = Field(default="No description provided.", description="Brief document description")
    author: Optional[str] = Field(default="Anonymous", description="Author of the document")
    tags: Optional[List[str]] = Field(default_factory=lambda: ["tag1", "tag2", "tag3"], description="Tags for categorization")

    class Config:
        orm_mode = True  # Enables compatibility with ORMs like SQLAlchemy


if __name__ == "__main__":
    print("Ready Player 1")
    # Test the MetaDataModel
    metadata = MetaDataModel(
        document_id="12345",
        title="My First Document"
    )
    logger.debug(metadata.model_dump_json(indent=4))    
    logger.debug("Done")