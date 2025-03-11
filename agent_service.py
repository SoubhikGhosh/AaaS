from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from sqlalchemy import create_engine, Column, Integer, String, JSON, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import httpx
import os
import json
import datetime
import logging
import uuid
import importlib.util
import sys
from pathlib import Path
import asyncio
import hashlib
import time
import re
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent_service.log")
    ]
)
logger = logging.getLogger("agent_service")

# Database setup - PostgreSQL with your specific configuration
DB_CONFIG = {
    'dbname': 'agents',     # Changed from 'shop_data' to 'agents'
    'user': 'soubhikghosh',
    'password': '99ghosh', 
    'host': '127.0.0.1',
    'port': '5432'
}

# Allow overriding with a full DATABASE_URL if preferred
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"

# We'll initialize the engine in startup_event after ensuring database exists
engine = None
SessionLocal = None
Base = declarative_base()

# Dependency to get DB session
def get_db():
    # Ensure SessionLocal is initialized
    if 'SessionLocal' not in globals() or SessionLocal is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection not initialized"
        )
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Database models
class AgentDB(Base):
    __tablename__ = "agents"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    specifications = Column(JSON)
    validations = Column(JSON)
    is_active = Column(Boolean, default=True)
    code = Column(Text)
    version = Column(Integer, default=1)
    execution_count = Column(Integer, default=0)
    average_execution_time_ms = Column(Integer, default=0)

# Pydantic models
class ValidationRule(BaseModel):
    field: str
    rule_type: str  # one of "type", "range", "regex", "required", "enum", "custom"
    parameters: Dict[str, Any] = {}
    error_message: Optional[str] = None

class AgentSpecification(BaseModel):
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    dependencies: List[str] = []
    timeout_seconds: int = 30
    memory_required_mb: int = 128
    description: str = ""
    tags: List[str] = []
    example_inputs: List[Dict[str, Any]] = []
    example_outputs: List[Dict[str, Any]] = []

class AgentCreate(BaseModel):
    name: str
    description: str
    specifications: AgentSpecification
    validations: List[ValidationRule] = []

    @validator('name')
    def name_must_be_valid(cls, v):
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Name must be alphanumeric with optional underscores')
        return v

class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    specifications: Optional[AgentSpecification] = None
    validations: Optional[List[ValidationRule]] = None
    is_active: Optional[bool] = None

    @validator('name')
    def name_must_be_valid(cls, v):
        if v is not None and not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Name must be alphanumeric with optional underscores')
        return v

class AgentResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    specifications: AgentSpecification
    validations: List[ValidationRule]
    is_active: bool
    version: int
    execution_count: Optional[int] = 0
    average_execution_time_ms: Optional[int] = 0

class AgentExecutionRequest(BaseModel):
    input_data: Dict[str, Any]

class AgentExecutionResponse(BaseModel):
    output: Dict[str, Any]
    execution_time_ms: float
    agent_id: str
    agent_name: str
    agent_version: int

class AgentCodeResponse(BaseModel):
    code: str

# Helper function to create database if it doesn't exist
def create_database_if_not_exists(config):
    """
    Create the database if it doesn't exist.
    Returns True if database was created, False if it already existed.
    """
    # Connect to the default 'postgres' database to check if our target DB exists
    default_db_url = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/postgres"
    logger.info(f"db url -> {default_db_url}")
    
    try:
        # Add this import
        from sqlalchemy import text
        
        # Create a temporary connection to check if our database exists
        temp_engine = create_engine(default_db_url)
        with temp_engine.connect() as conn:
            # Don't autocommit to allow rollback if needed
            conn.execution_options(isolation_level="AUTOCOMMIT")
            
            # These two lines need to be changed - wrap the SQL with text()
            result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{config['dbname']}'"))
            exists = result.scalar() == 1
            
            if not exists:
                logger.info(f"Creating database '{config['dbname']}'")
                # This line also needs to be changed
                conn.execute(text(f"CREATE DATABASE {config['dbname']}"))
                return True
            
            return False
    except Exception as e:
        logger.error(f"Error checking/creating database: {e}")
        raise

# Service classes
class GeminiService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set")
        
        # Configure the Google Generative AI client
        genai.configure(api_key=self.api_key)
        
        # Create the model with specified configuration
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",  # Use an available model based on your API access
            safety_settings=[
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            ],
            generation_config={"temperature": 0.7, "top_p": 0.95, "top_k": 40}
        )
        
        self._cache = {}  # Simple in-memory cache
        self._rate_limit_semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent requests
    
    async def generate_code(self, prompt: str, cache_key: Optional[str] = None) -> str:
        """Generate code using Gemini API with caching and rate limiting"""
        # Use cache if available
        if cache_key and cache_key in self._cache:
            logger.info(f"Using cached result for key: {cache_key}")
            return self._cache[cache_key]
        
        logger.info(f"Generating code with prompt length: {len(prompt)}")
        
        # Rate limiting using semaphore
        async with self._rate_limit_semaphore:
            # Additional time-based rate limiting
            await self._apply_rate_limit()
            
            try:
                # Use asyncio.to_thread to run the synchronous genai call in a separate thread
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    prompt
                )
                
                # Extract the generated text
                generated_text = response.text
                
                # Extract code between ```python and ``` markers
                code_start = generated_text.find("```python")
                if code_start != -1:
                    code_start += 9  # Length of "```python"
                    code_end = generated_text.find("```", code_start)
                    if code_end != -1:
                        code = generated_text[code_start:code_end].strip()
                    else:
                        code = generated_text[code_start:].strip()
                else:
                    # If no python markers, try to extract any code block
                    code_start = generated_text.find("```")
                    if code_start != -1:
                        code_start += 3  # Length of "```"
                        code_end = generated_text.find("```", code_start)
                        if code_end != -1:
                            code = generated_text[code_start:code_end].strip()
                        else:
                            code = generated_text[code_start:].strip()
                    else:
                        # If no code blocks, use the entire text
                        code = generated_text.strip()
                
                # Store in cache if cache_key provided
                if cache_key:
                    self._cache[cache_key] = code
                
                return code
                
            except Exception as e:
                logger.error(f"Gemini API error: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Error from Gemini API: {str(e)}"
                )
    
    # Track request times for rate limiting
    _last_request_times = []
    
    async def _apply_rate_limit(self):
        """Advanced rate limiting implementation"""
        now = time.time()
        
        # Clean up old requests (older than 60 seconds)
        self._last_request_times = [t for t in self._last_request_times if now - t < 60]
        
        # Check if we've made more than 60 requests in the last 60 seconds
        if len(self._last_request_times) >= 60:
            # Wait until we're under the limit
            oldest_time = min(self._last_request_times)
            wait_time = 60 - (now - oldest_time)
            if wait_time > 0:
                logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # Add current request time
        self._last_request_times.append(time.time())

class PromptGeneratorService:
    @staticmethod
    def generate_agent_prompt(agent_data: AgentCreate) -> str:
        """Generate a prompt for the Gemini API based on agent specifications"""
        prompt = f"""
# Agent Code Generation Task

Create a Python function for an agent with the following specifications:

## Agent Name
{agent_data.name}

## Description
{agent_data.description}

## Input Schema
```json
{json.dumps(agent_data.specifications.input_schema, indent=2)}
```

## Output Schema
```json
{json.dumps(agent_data.specifications.output_schema, indent=2)}
```

## Dependencies
{', '.join(agent_data.specifications.dependencies) if agent_data.specifications.dependencies else 'No specific dependencies'}

## Resource Requirements
- Timeout: {agent_data.specifications.timeout_seconds} seconds
- Memory: {agent_data.specifications.memory_required_mb} MB

## Validation Rules
"""
        
        if agent_data.validations:
            for i, validation in enumerate(agent_data.validations, 1):
                prompt += f"""
{i}. Field: {validation.field}
   Rule Type: {validation.rule_type}
   Parameters: {json.dumps(validation.parameters)}
   Error Message: {validation.error_message or 'Default error message'}
"""
        else:
            prompt += "No specific validation rules.\n"
        
        prompt += """
## Example Inputs and Expected Outputs
"""
        
        for i, (example_input, example_output) in enumerate(zip(
            agent_data.specifications.example_inputs,
            agent_data.specifications.example_outputs
        ), 1):
            prompt += f"""
Example {i}:
Input: {json.dumps(example_input, indent=2)}
Expected Output: {json.dumps(example_output, indent=2)}
"""
        
        prompt += """
## Code Requirements
1. Create a Python function named `process` that takes the input as defined in the schema and returns output matching the output schema.
2. Implement all validation rules as specified.
3. Handle errors gracefully and return appropriate error messages.
4. Include docstrings, type hints, and comments.
5. Follow PEP 8 style guidelines.
6. The code should be efficient and handle edge cases.
7. Include any necessary imports at the top of the file.
8. Ensure the code is production-ready and properly tested.

Please return only the Python code, starting with imports and ending with the function implementation. Do not include explanations, markdown formatting, or any text outside of the actual Python code.
"""
        
        return prompt

    @staticmethod
    def generate_cache_key(agent_data: Union[AgentCreate, AgentUpdate]) -> str:
        """Generate a cache key for the agent data to avoid redundant API calls"""
        data_dict = agent_data.dict(exclude_unset=True)
        serialized = json.dumps(data_dict, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()

class AgentService:
    def __init__(self, db: Session, gemini_service: GeminiService, prompt_service: PromptGeneratorService):
        self.db = db
        self.gemini_service = gemini_service
        self.prompt_service = prompt_service
        self.agents_dir = Path("./agents")
        self.agents_dir.mkdir(exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=10)
    
    async def create_agent(self, agent_data: AgentCreate) -> AgentResponse:
        """Create a new agent"""
        # Check if agent with this name already exists
        existing_agent = self.db.query(AgentDB).filter(AgentDB.name == agent_data.name).first()
        if existing_agent:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Agent with name '{agent_data.name}' already exists"
            )
        
        # Generate prompt for Gemini
        prompt = self.prompt_service.generate_agent_prompt(agent_data)
        cache_key = self.prompt_service.generate_cache_key(agent_data)
        
        # Generate code using Gemini
        try:
            code = await self.gemini_service.generate_code(prompt, cache_key)
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate agent code: {str(e)}"
            )
        
        # Create a new agent in the database
        agent_id = str(uuid.uuid4())
        db_agent = AgentDB(
            id=agent_id,
            name=agent_data.name,
            description=agent_data.description,
            specifications=agent_data.specifications.dict(),
            validations=[v.dict() for v in agent_data.validations],
            code=code,
            created_at=datetime.datetime.utcnow(),
            updated_at=datetime.datetime.utcnow(),
            is_active=True,
            version=1
        )
        
        self.db.add(db_agent)
        
        try:
            self.db.commit()
            self.db.refresh(db_agent)
        except Exception as e:
            self.db.rollback()
            logger.error(f"Database error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save agent: {str(e)}"
            )
        
        # Save the code to a file asynchronously
        asyncio.create_task(self._save_agent_code_to_file(agent_id, agent_data.name, code))
        
        return AgentResponse(
            id=db_agent.id,
            name=db_agent.name,
            description=db_agent.description,
            created_at=db_agent.created_at,
            updated_at=db_agent.updated_at,
            specifications=AgentSpecification(**db_agent.specifications),
            validations=[ValidationRule(**v) for v in db_agent.validations],
            is_active=db_agent.is_active,
            version=db_agent.version,
            execution_count=db_agent.execution_count,
            average_execution_time_ms=db_agent.average_execution_time_ms
        )
    
    async def update_agent(self, agent_id: str, agent_data: AgentUpdate) -> AgentResponse:
        """Update an existing agent"""
        db_agent = self.db.query(AgentDB).filter(AgentDB.id == agent_id).first()
        if not db_agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID '{agent_id}' not found"
            )
        
        update_data = agent_data.dict(exclude_unset=True)
        
        # If updating name, check for duplicates
        if "name" in update_data and update_data["name"] != db_agent.name:
            existing = self.db.query(AgentDB).filter(AgentDB.name == update_data["name"]).first()
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Agent with name '{update_data['name']}' already exists"
                )
        
        # If updating specifications or validations, generate new code
        if "specifications" in update_data or "validations" in update_data:
            # Prepare full agent data for code generation
            current_specs = AgentSpecification(**db_agent.specifications)
            current_validations = [ValidationRule(**v) for v in db_agent.validations]
            
            full_agent_data = AgentCreate(
                name=update_data.get("name", db_agent.name),
                description=update_data.get("description", db_agent.description),
                specifications=update_data.get("specifications", current_specs),
                validations=update_data.get("validations", current_validations)
            )
            
            # Generate new prompt and code
            prompt = self.prompt_service.generate_agent_prompt(full_agent_data)
            cache_key = self.prompt_service.generate_cache_key(full_agent_data)
            
            try:
                code = await self.gemini_service.generate_code(prompt, cache_key)
                update_data["code"] = code
                update_data["version"] = db_agent.version + 1
            except Exception as e:
                logger.error(f"Error generating updated code: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to generate updated agent code: {str(e)}"
                )
            
            # Save the new code to a file asynchronously
            asyncio.create_task(self._save_agent_code_to_file(
                agent_id, 
                update_data.get("name", db_agent.name),
                code
            ))
        
        # Update the database record
        for key, value in update_data.items():
            if key in ["specifications", "validations"]:
                # Convert Pydantic models to dictionaries for JSON storage
                if hasattr(value, "dict"):
                    setattr(db_agent, key, value.dict())
                else:
                    setattr(db_agent, key, [v.dict() if hasattr(v, "dict") else v for v in value])
            else:
                setattr(db_agent, key, value)
        
        db_agent.updated_at = datetime.datetime.utcnow()
        
        try:
            self.db.commit()
            self.db.refresh(db_agent)
        except Exception as e:
            self.db.rollback()
            logger.error(f"Database error during update: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update agent: {str(e)}"
            )
        
        return AgentResponse(
            id=db_agent.id,
            name=db_agent.name,
            description=db_agent.description,
            created_at=db_agent.created_at,
            updated_at=db_agent.updated_at,
            specifications=AgentSpecification(**db_agent.specifications),
            validations=[ValidationRule(**v) for v in db_agent.validations],
            is_active=db_agent.is_active,
            version=db_agent.version,
            execution_count=db_agent.execution_count,
            average_execution_time_ms=db_agent.average_execution_time_ms
        )
    
    def delete_agent(self, agent_id: str) -> None:
        """Delete an agent"""
        db_agent = self.db.query(AgentDB).filter(AgentDB.id == agent_id).first()
        if not db_agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID '{agent_id}' not found"
            )
        
        # Soft delete by setting is_active to False
        db_agent.is_active = False
        db_agent.updated_at = datetime.datetime.utcnow()
        
        try:
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            logger.error(f"Database error during delete: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete agent: {str(e)}"
            )
    
    def get_agent(self, agent_id: str) -> AgentResponse:
        """Get an agent by ID"""
        db_agent = self.db.query(AgentDB).filter(AgentDB.id == agent_id).first()
        if not db_agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID '{agent_id}' not found"
            )
        
        return AgentResponse(
            id=db_agent.id,
            name=db_agent.name,
            description=db_agent.description,
            created_at=db_agent.created_at,
            updated_at=db_agent.updated_at,
            specifications=AgentSpecification(**db_agent.specifications),
            validations=[ValidationRule(**v) for v in db_agent.validations],
            is_active=db_agent.is_active,
            version=db_agent.version,
            execution_count=db_agent.execution_count,
            average_execution_time_ms=db_agent.average_execution_time_ms
        )
    
    def list_agents(self, skip: int = 0, limit: int = 100, include_inactive: bool = False) -> List[AgentResponse]:
        """List all agents"""
        query = self.db.query(AgentDB)
        if not include_inactive:
            query = query.filter(AgentDB.is_active == True)
        
        db_agents = query.order_by(AgentDB.updated_at.desc()).offset(skip).limit(limit).all()
        
        return [
            AgentResponse(
                id=db_agent.id,
                name=db_agent.name,
                description=db_agent.description,
                created_at=db_agent.created_at,
                updated_at=db_agent.updated_at,
                specifications=AgentSpecification(**db_agent.specifications),
                validations=[ValidationRule(**v) for v in db_agent.validations],
                is_active=db_agent.is_active,
                version=db_agent.version,
                execution_count=db_agent.execution_count,
                average_execution_time_ms=db_agent.average_execution_time_ms
            )
            for db_agent in db_agents
        ]
    
    async def execute_agent(self, agent_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an agent with the given input data"""
        db_agent = self.db.query(AgentDB).filter(AgentDB.id == agent_id).first()
        if not db_agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID '{agent_id}' not found"
            )
        
        if not db_agent.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Agent '{db_agent.name}' is inactive"
            )
        
        # Load the agent module
        try:
            agent_module = await self._load_agent_module(agent_id, db_agent.name, db_agent.code)
        except Exception as e:
            logger.error(f"Error loading agent module: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load agent code: {str(e)}"
            )
        
        # Execute the agent
        start_time = time.time()
        try:
            # Run in threadpool to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self._executor, 
                agent_module.process,
                input_data
            )
            
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Update execution statistics
            self._update_execution_stats(db_agent, execution_time)
            
            return {
                "output": result,
                "execution_time_ms": execution_time,
                "agent_id": agent_id,
                "agent_name": db_agent.name,
                "agent_version": db_agent.version
            }
        except Exception as e:
            logger.error(f"Error executing agent: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error executing agent: {str(e)}"
            )
    
    def _update_execution_stats(self, db_agent: AgentDB, execution_time: float) -> None:
        """Update agent execution statistics"""
        # Update execution count and average execution time
        total_execution_time = db_agent.average_execution_time_ms * db_agent.execution_count
        db_agent.execution_count += 1
        db_agent.average_execution_time_ms = int((total_execution_time + execution_time) / db_agent.execution_count)
        
        try:
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating execution stats: {e}")
            # Don't raise an exception, as this is a non-critical operation
    
    async def _save_agent_code_to_file(self, agent_id: str, agent_name: str, code: str) -> None:
        """Save agent code to a file"""
        # Create a secure filename from the agent name
        safe_name = ''.join(c if c.isalnum() else '_' for c in agent_name)
        file_path = self.agents_dir / f"{agent_id}_{safe_name}.py"
        
        # Write code to file
        try:
            async def write_file():
                with open(file_path, "w") as f:
                    f.write(code)
            
            await asyncio.to_thread(write_file)
            logger.info(f"Saved agent code to {file_path}")
        except Exception as e:
            logger.error(f"Error saving agent code to file: {e}")
            # We continue even if file saving fails
    
    async def _load_agent_module(self, agent_id: str, agent_name: str, code: str) -> Any:
        """Load agent code as a Python module"""
        # Create a secure filename from the agent name
        safe_name = ''.join(c if c.isalnum() else '_' for c in agent_name)
        module_name = f"agent_{agent_id}_{safe_name}"
        file_path = self.agents_dir / f"{agent_id}_{safe_name}.py"
        
        # Ensure the file exists
        if not file_path.exists():
            # Create the file if it doesn't exist
            await self._save_agent_code_to_file(agent_id, agent_name, code)
        
        # Load the module
        try:
            # Clear module from cache if it's already loaded
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                raise ImportError(f"Could not load spec for {module_name} from {file_path}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Verify that the module has a process function
            if not hasattr(module, "process") or not callable(module.process):
                raise AttributeError(f"Module {module_name} does not have a callable 'process' function")
            
            return module
        except Exception as e:
            logger.error(f"Error loading agent module: {e}")
            raise

# Create FastAPI app
app = FastAPI(
    title="Agent as a Service",
    description="A service for dynamically creating and managing AI agents using Gemini API",
    version="1.0.0"
)

# Add CORS middleware - allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins as requested
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection
def get_gemini_service():
    return GeminiService()

def get_prompt_service():
    return PromptGeneratorService()

def get_agent_service(db: Session = Depends(get_db), 
                     gemini_service: GeminiService = Depends(get_gemini_service),
                     prompt_service: PromptGeneratorService = Depends(get_prompt_service)):
    return AgentService(db, gemini_service, prompt_service)

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.datetime.utcnow(),
        "version": "1.0.0"
    }

# Agent endpoints
@app.post("/agents", response_model=AgentResponse, status_code=status.HTTP_201_CREATED, tags=["Agents"])
async def create_agent(
    agent_data: AgentCreate,
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    Create a new agent with the provided specifications.
    
    This will generate code using Gemini API based on the agent specifications
    and validations provided.
    """
    return await agent_service.create_agent(agent_data)

@app.get("/agents", response_model=List[AgentResponse], tags=["Agents"])
async def list_agents(
    skip: int = 0,
    limit: int = 100,
    include_inactive: bool = False,
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    List all agents with pagination support.
    
    By default, only active agents are returned.
    Set `include_inactive=true` to include inactive agents.
    """
    return agent_service.list_agents(skip, limit, include_inactive)

@app.get("/agents/{agent_id}", response_model=AgentResponse, tags=["Agents"])
async def get_agent(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    Get details of a specific agent by ID.
    """
    return agent_service.get_agent(agent_id)

@app.put("/agents/{agent_id}", response_model=AgentResponse, tags=["Agents"])
async def update_agent(
    agent_id: str,
    agent_data: AgentUpdate,
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    Update an existing agent.
    
    If specifications or validations are updated, new code will be generated.
    """
    return await agent_service.update_agent(agent_id, agent_data)

@app.delete("/agents/{agent_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Agents"])
async def delete_agent(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    Delete (or deactivate) an agent.
    
    This performs a soft delete by marking the agent as inactive.
    """
    agent_service.delete_agent(agent_id)
    return None

@app.post("/agents/{agent_id}/execute", response_model=AgentExecutionResponse, tags=["Execution"])
async def execute_agent(
    agent_id: str,
    request: AgentExecutionRequest,
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    Execute an agent with the provided input data.
    
    Returns the output of the agent's processing along with execution metrics.
    """
    result = await agent_service.execute_agent(agent_id, request.input_data)
    return AgentExecutionResponse(**result)

@app.get("/agents/{agent_id}/code", response_model=AgentCodeResponse, tags=["Agents"])
async def get_agent_code(
    agent_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the code for a specific agent.
    """
    db_agent = db.query(AgentDB).filter(AgentDB.id == agent_id).first()
    if not db_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with ID '{agent_id}' not found"
        )
    
    return AgentCodeResponse(code=db_agent.code)

@app.post("/agents/{agent_id}/activate", response_model=AgentResponse, tags=["Agents"])
async def activate_agent(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service),
    db: Session = Depends(get_db)
):
    """
    Activate a previously deactivated agent.
    """
    db_agent = db.query(AgentDB).filter(AgentDB.id == agent_id).first()
    if not db_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with ID '{agent_id}' not found"
        )
    
    if db_agent.is_active:
        return agent_service.get_agent(agent_id)
    
    db_agent.is_active = True
    db_agent.updated_at = datetime.datetime.utcnow()
    
    try:
        db.commit()
        db.refresh(db_agent)
    except Exception as e:
        db.rollback()
        logger.error(f"Database error during activation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to activate agent: {str(e)}"
        )
    
    return agent_service.get_agent(agent_id)

# Metrics and monitoring endpoints
@app.get("/metrics", tags=["Monitoring"])
async def get_metrics(db: Session = Depends(get_db)):
    """
    Get service metrics.
    
    This includes counts of agents, executions, etc.
    """
    try:
        total_agents = db.query(AgentDB).count()
        active_agents = db.query(AgentDB).filter(AgentDB.is_active == True).count()
        total_executions = db.query(AgentDB).with_entities(AgentDB.execution_count).all()
        total_executions_count = sum(count for (count,) in total_executions)
        
        # Get agents with most executions
        top_agents = db.query(AgentDB).filter(AgentDB.is_active == True).order_by(AgentDB.execution_count.desc()).limit(5).all()
        top_agents_data = [
            {
                "id": agent.id,
                "name": agent.name,
                "executions": agent.execution_count,
                "avg_execution_time_ms": agent.average_execution_time_ms
            }
            for agent in top_agents
        ]
        
        return {
            "agents": {
                "total": total_agents,
                "active": active_agents,
                "inactive": total_agents - active_agents
            },
            "executions": {
                "total": total_executions_count,
                "top_agents": top_agents_data
            },
            "system": {
                "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, "start_time") else 0,
                "version": "1.0.0"
            }
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    logger.info("Agent Service starting up")
    app.state.start_time = time.time()
    
    # Create agents directory if it doesn't exist
    Path("./agents").mkdir(exist_ok=True)
    
    # Verify Gemini API key is set
    api_key = os.getenv("GEMINI_API_KEY", "AIzaSyCD6DGeERwWQbBC6BK1Hq0ecagQj72rqyQ")  # Use default API key if not set
    if not api_key:
        logger.warning("GEMINI_API_KEY environment variable not set. Using default API key.")
    
    # Pre-configure the Gemini API client
    try:
        genai.configure(api_key=api_key)
        logger.info("Gemini API client configured successfully")
    except Exception as e:
        logger.error(f"Error configuring Gemini API client: {e}")
    
    # Parse database URL if provided, otherwise use components
    global engine, SessionLocal
    
    if DATABASE_URL.startswith("postgresql"):
        # Extract database name from URL for checking existence
        db_name = DATABASE_URL.split("/")[-1].split("?")[0]
        
        # Extract components from the DATABASE_URL if it was provided
        db_config_for_check = DB_CONFIG.copy()
        if os.getenv("DATABASE_URL"):
            from urllib.parse import urlparse
            parsed_url = urlparse(DATABASE_URL)
            db_config_for_check = {
                'user': parsed_url.username or DB_CONFIG['user'],
                'password': parsed_url.password or DB_CONFIG['password'],
                'host': parsed_url.hostname or DB_CONFIG['host'],
                'port': str(parsed_url.port) if parsed_url.port else DB_CONFIG['port'],
                'dbname': db_name
            }
        
        try:
            # Attempt to create database if needed
            created = create_database_if_not_exists(db_config_for_check)
            if created:
                logger.info(f"Created database: {db_config_for_check['dbname']}")
        except Exception as e:
            logger.error(f"Error during database creation: {e}")
            logger.warning("Continuing with startup, but database operations may fail")
    
    # Now create engine using the full URL (database should exist at this point)
    engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    await asyncio.create_task(setup_langchain())

    # Create tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created or verified")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        logger.warning("Application may not function correctly without required database tables")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Agent Service shutting down")

@app.get("/db-check", tags=["Diagnostics"])
async def check_database_connection():
    """
    Check database connection and return detailed diagnostic information.
    """
    results = {
        "config": {
            "dbname": DB_CONFIG['dbname'],
            "user": DB_CONFIG['user'],
            "host": DB_CONFIG['host'],
            "port": DB_CONFIG['port'],
            "connection_string": DATABASE_URL
        },
        "status": "unknown"
    }
    
    # Test direct PostgreSQL connection
    try:
        import psycopg2
        conn = psycopg2.connect(
            dbname=DB_CONFIG['dbname'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port']
        )
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        results["direct_connection"] = "successful"
    except Exception as e:
        results["direct_connection"] = f"failed: {str(e)}"
    
    # Test SQLAlchemy connection
    try:
        from sqlalchemy import create_engine, text
        test_engine = create_engine(DATABASE_URL)
        with test_engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        results["sqlalchemy_connection"] = "successful"
    except Exception as e:
        results["sqlalchemy_connection"] = f"failed: {str(e)}"
    
    return results

async def setup_langchain():
    import langchain_integration
    try:
        langchain_integration.integrate_with_main_app(app)
        logger.info("LangChain integration initialized successfully")
    except Exception as e:
        logger.error(f"Langchain Initiation error: {e}")

# Main entry point
if __name__ == "__main__":
    import uvicorn
    # setup_langchain()
    logger.info("Starting Agent Service")
    uvicorn.run(
        "agent_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=4  # Adjust based on available CPU cores
    )