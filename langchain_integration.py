from typing import Dict, List, Any, Optional, Union, Tuple
import json
import os
from pathlib import Path
import asyncio
from datetime import datetime
import logging
import time
import uuid

# Import from main application
# from agent_service import AgentDB, get_db, AgentResponse, ValidationRule, AgentSpecification

# FastAPI imports
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

# LangChain imports - older versions that don't require Rust
from langchain.llms import GooglePalm
from langchain.chat_models import ChatGooglePalm
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.agents import Tool, AgentExecutor, initialize_agent, AgentType
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.memory import ConversationBufferMemory

# Configure logging
logger = logging.getLogger("langchain_integration")

def get_db():
    from agent_service import get_db as get_db_func
    return next(get_db_func())

def get_agent_db_class():
    from agent_service import AgentDB
    return AgentDB

# LangChain Models
class LangChainModelService:
    def __init__(self, api_key=None):
        """Initialize LangChain model service with Google API"""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "AIzaSyCD6DGeERwWQbBC6BK1Hq0ecagQj72rqyQ")
        
        # Create Google model
        self.llm = ChatGooglePalm(
            google_api_key=self.api_key,
            temperature=0.7,
            model_name="models/gemini-pro"  # Use an available model
        )
    
    def get_llm(self):
        """Get the LangChain LLM"""
        return self.llm

# Validation Tools
class ValidationToolsFactory:
    @staticmethod
    def create_type_validation_tool(field_name: str, expected_type: str) -> Tool:
        """Create a tool to validate data types"""
        def validate_type(value: Any) -> str:
            if expected_type == "string" and not isinstance(value, str):
                return f"Validation failed: {field_name} must be a string"
            elif expected_type == "number" and not isinstance(value, (int, float)):
                return f"Validation failed: {field_name} must be a number"
            elif expected_type == "boolean" and not isinstance(value, bool):
                return f"Validation failed: {field_name} must be a boolean"
            elif expected_type == "array" and not isinstance(value, list):
                return f"Validation failed: {field_name} must be an array"
            elif expected_type == "object" and not isinstance(value, dict):
                return f"Validation failed: {field_name} must be an object"
            return f"Validation passed: {field_name} is a valid {expected_type}"
        
        return Tool(
            name=f"validate_{field_name}_type",
            description=f"Validate that {field_name} is of type {expected_type}",
            func=validate_type
        )
    
    @staticmethod
    def create_range_validation_tool(field_name: str, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Tool:
        """Create a tool to validate numeric ranges"""
        def validate_range(value: Any) -> str:
            if not isinstance(value, (int, float)):
                return f"Validation failed: {field_name} must be a number for range validation"
            
            if min_val is not None and value < min_val:
                return f"Validation failed: {field_name} must be at least {min_val}"
            
            if max_val is not None and value > max_val:
                return f"Validation failed: {field_name} must be at most {max_val}"
            
            range_desc = []
            if min_val is not None:
                range_desc.append(f">= {min_val}")
            if max_val is not None:
                range_desc.append(f"<= {max_val}")
            
            return f"Validation passed: {field_name} is within range ({', '.join(range_desc)})"
        
        range_desc = []
        if min_val is not None:
            range_desc.append(f">= {min_val}")
        if max_val is not None:
            range_desc.append(f"<= {max_val}")
        
        return Tool(
            name=f"validate_{field_name}_range",
            description=f"Validate that {field_name} is within range ({', '.join(range_desc)})",
            func=validate_range
        )
    
    @staticmethod
    def create_regex_validation_tool(field_name: str, pattern: str) -> Tool:
        """Create a tool to validate against regex patterns"""
        import re
        
        def validate_regex(value: str) -> str:
            if not isinstance(value, str):
                return f"Validation failed: {field_name} must be a string for regex validation"
            
            if re.match(pattern, value):
                return f"Validation passed: {field_name} matches the required pattern"
            else:
                return f"Validation failed: {field_name} must match pattern {pattern}"
        
        return Tool(
            name=f"validate_{field_name}_pattern",
            description=f"Validate that {field_name} matches regex pattern {pattern}",
            func=validate_regex
        )
    
    @staticmethod
    def create_enum_validation_tool(field_name: str, allowed_values: List[Any]) -> Tool:
        """Create a tool to validate against enum values"""
        def validate_enum(value: Any) -> str:
            if value in allowed_values:
                return f"Validation passed: {field_name} has a valid value from the allowed list"
            else:
                return f"Validation failed: {field_name} must be one of {allowed_values}"
        
        return Tool(
            name=f"validate_{field_name}_enum",
            description=f"Validate that {field_name} is one of these values: {allowed_values}",
            func=validate_enum
        )

# LangChain Agent Creator
class LangChainAgentFactory:
    def __init__(self, model_service: LangChainModelService):
        self.model_service = model_service
        self.llm = model_service.get_llm()
        self.validation_tools_factory = ValidationToolsFactory()
    
    def create_tools_from_validations(self, validations: List[Dict[str, Any]]) -> List[Tool]:
        """Create LangChain tools from validation rules"""
        tools = []
        
        for validation in validations:
            field = validation.get("field", "")
            rule_type = validation.get("rule_type", "")
            parameters = validation.get("parameters", {})
            
            if rule_type == "type":
                expected_type = parameters.get("type", "string")
                tools.append(self.validation_tools_factory.create_type_validation_tool(field, expected_type))
            
            elif rule_type == "range":
                min_val = parameters.get("min")
                max_val = parameters.get("max")
                tools.append(self.validation_tools_factory.create_range_validation_tool(field, min_val, max_val))
            
            elif rule_type == "regex":
                pattern = parameters.get("pattern", "")
                if pattern:
                    tools.append(self.validation_tools_factory.create_regex_validation_tool(field, pattern))
            
            elif rule_type == "enum":
                allowed_values = parameters.get("values", [])
                if allowed_values:
                    tools.append(self.validation_tools_factory.create_enum_validation_tool(field, allowed_values))
        
        return tools
    
    def create_agent(self, 
                    name: str, 
                    description: str, 
                    input_schema: Dict[str, Any], 
                    output_schema: Dict[str, Any],
                    validations: List[Dict[str, Any]] = None) -> AgentExecutor:
        """Create a LangChain agent with validations"""
        # Create tools from validations
        tools = []
        if validations:
            tools = self.create_tools_from_validations(validations)
        
        # Create a system prompt
        system_template = f"""You are an AI agent named {name}.
Your purpose is: {description}

You receive input with this schema:
```json
{json.dumps(input_schema, indent=2)}
```

You must produce output with this schema:
```json
{json.dumps(output_schema, indent=2)}
```

Validate all inputs carefully before processing.
Your output must strictly follow the output schema.
"""
        
        # Create a memory
        memory = ConversationBufferMemory(memory_key="chat_history")
        
        # Create output parser
        response_schemas = []
        for key, value in output_schema.get("properties", {}).items():
            schema_type = value.get("type", "string")
            schema_desc = value.get("description", f"The {key}")
            response_schemas.append(ResponseSchema(name=key, description=schema_desc, type=schema_type))
        
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        
        # Define a prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{input}\n{format_instructions}")
        ])
        
        # Create chain or agent based on whether tools are available
        if tools:
            # Create agent with tools
            agent = initialize_agent(
                tools=tools,
                llm=self.llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                memory=memory,
                handle_parsing_errors=True
            )
            return agent
        else:
            # Create a simple chain without tools
            format_instructions = output_parser.get_format_instructions()
            
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                verbose=True,
                memory=memory
            )
            
            # Wrap the chain to handle the output parsing
            def process_input(input_data):
                result = chain.run(input=json.dumps(input_data), format_instructions=format_instructions)
                try:
                    return output_parser.parse(result)
                except Exception as e:
                    # If parsing fails, return the raw result
                    return {"error": str(e), "raw_output": result}
            
            return process_input

# LangChain Agent Service
class LangChainAgentService:
    def __init__(self, db_session, model_service: LangChainModelService = None):
        """Initialize the LangChain agent service"""
        self.db = db_session
        self.model_service = model_service or LangChainModelService()
        self.agent_factory = LangChainAgentFactory(self.model_service)
        self.loaded_agents = {}  # Cache for loaded agents
    
    async def create_agent_from_specifications(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a LangChain agent from specifications"""
        name = agent_data.get("name", "")
        description = agent_data.get("description", "")
        specifications = agent_data.get("specifications", {})
        validations = agent_data.get("validations", [])
        
        input_schema = specifications.get("input_schema", {})
        output_schema = specifications.get("output_schema", {})
        
        # Create the agent
        agent = self.agent_factory.create_agent(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            validations=validations
        )
        
        # Return the agent configuration
        return {
            "name": name,
            "agent": agent,
            "input_schema": input_schema,
            "output_schema": output_schema
        }
    
    async def execute_agent(self, agent_config: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a LangChain agent with input data"""
        agent = agent_config.get("agent")
        
        if callable(agent):
            # For simple chains wrapped as callable
            return await asyncio.to_thread(agent, input_data)
        else:
            # For AgentExecutor instances
            return await asyncio.to_thread(agent.run, input_data)

# FastAPI router and endpoints
langchain_router = APIRouter(prefix="/langchain", tags=["LangChain"])

# Pydantic models for API
class LangChainToolConfig(BaseModel):
    name: str
    description: str
    type: str  # "type", "range", "regex", "enum"
    parameters: Dict[str, Any] = Field(default_factory=dict)

class LangChainAgentConfig(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    tools: List[LangChainToolConfig] = Field(default_factory=list)

class LangChainExecutionRequest(BaseModel):
    input_data: Dict[str, Any]

class LangChainExecutionResponse(BaseModel):
    output: Dict[str, Any]
    execution_time_ms: float
    agent_id: str
    agent_name: str

# Dependency to get LangChain service
def get_langchain_service(db = Depends(get_db)):
    model_service = LangChainModelService()
    return LangChainAgentService(db, model_service)

# Endpoints
@langchain_router.post("/agents", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_langchain_agent(
    config: LangChainAgentConfig,
    db = Depends(get_db),
    langchain_service: LangChainAgentService = Depends(get_langchain_service)
):
    """Create a new LangChain agent"""
    try:
        # Convert tools to validations
        validations = []
        for tool in config.tools:
            validations.append({
                "field": tool.name,
                "rule_type": tool.type,
                "parameters": tool.parameters,
                "error_message": tool.description
            })
        
        # Create agent data
        agent_data = {
            "name": config.name,
            "description": config.description,
            "specifications": {
                "input_schema": config.input_schema,
                "output_schema": config.output_schema
            },
            "validations": validations
        }
        
        # Create agent
        agent_config = await langchain_service.create_agent_from_specifications(agent_data)
        
        # Store in database
        agent_id = str(uuid.uuid4())
        AgentDB = get_agent_db_class()
        db_agent = AgentDB(
            id=agent_id,
            name=config.name,
            description=config.description,
            specifications=agent_data["specifications"],
            validations=validations,
            code=json.dumps(agent_data),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            is_active=True,
            version=1
        )
        
        db.add(db_agent)
        db.commit()
        db.refresh(db_agent)
        
        # Store in memory
        langchain_service.loaded_agents[agent_id] = agent_config
        
        return {
            "id": agent_id,
            "name": config.name,
            "description": config.description,
            "created_at": db_agent.created_at.isoformat(),
            "message": "LangChain agent created successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create LangChain agent: {str(e)}"
        )

@langchain_router.post("/agents/{agent_id}/execute", response_model=LangChainExecutionResponse)
async def execute_langchain_agent(
    agent_id: str,
    request: LangChainExecutionRequest,
    db = Depends(get_db),
    langchain_service: LangChainAgentService = Depends(get_langchain_service)
):
    """Execute a LangChain agent"""
    # Check if agent exists
    db_agent = db.query(AgentDB).filter(AgentDB.id == agent_id).first()
    if not db_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with ID {agent_id} not found"
        )
    
    if not db_agent.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent {db_agent.name} is inactive"
        )
    
    try:
        # Get agent from cache or create it
        agent_config = langchain_service.loaded_agents.get(agent_id)
        if not agent_config:
            # Load from database
            try:
                agent_data = json.loads(db_agent.code)
            except json.JSONDecodeError:
                # If the code is not JSON, create agent data from DB fields
                agent_data = {
                    "name": db_agent.name,
                    "description": db_agent.description,
                    "specifications": db_agent.specifications,
                    "validations": db_agent.validations
                }
            
            agent_config = await langchain_service.create_agent_from_specifications(agent_data)
            langchain_service.loaded_agents[agent_id] = agent_config
        
        # Execute the agent
        start_time = time.time()
        result = await langchain_service.execute_agent(agent_config, request.input_data)
        execution_time = (time.time() - start_time) * 1000
        
        # Update statistics
        db_agent.execution_count += 1
        total_time = db_agent.average_execution_time_ms * (db_agent.execution_count - 1)
        db_agent.average_execution_time_ms = int((total_time + execution_time) / db_agent.execution_count)
        db.commit()
        
        return {
            "output": result,
            "execution_time_ms": execution_time,
            "agent_id": agent_id,
            "agent_name": db_agent.name
        }
    except Exception as e:
        logger.error(f"Error executing LangChain agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing LangChain agent: {str(e)}"
        )

@langchain_router.get("/agents", response_model=List[Dict[str, Any]])
async def list_langchain_agents(
    skip: int = 0,
    limit: int = 100,
    db = Depends(get_db)
):
    """List all LangChain agents"""
    # For simplicity, we'll list all agents with 'code' field that can be parsed as JSON
    agents = []
    db_agents = db.query(AgentDB).filter(AgentDB.is_active == True).offset(skip).limit(limit).all()
    
    for agent in db_agents:
        try:
            json.loads(agent.code)  # Test if code is JSON
            agents.append({
                "id": agent.id,
                "name": agent.name,
                "description": agent.description,
                "created_at": agent.created_at.isoformat(),
                "updated_at": agent.updated_at.isoformat(),
                "execution_count": agent.execution_count,
                "average_execution_time_ms": agent.average_execution_time_ms,
                "is_langchain": True
            })
        except json.JSONDecodeError:
            # This is a regular agent, not a LangChain one
            continue
    
    return agents

# Function to integrate with main app
def integrate_with_main_app(app):
    """Add LangChain functionality to the main app"""
    # Include the router
    app.include_router(langchain_router)
    
    # Set up LangChain on startup
    @app.on_event("startup")
    async def setup_langchain():
        # Get a DB session
        from sqlalchemy.orm import Session
        from agent_service import SessionLocal
        
        db = SessionLocal()
        try:
            # Initialize the model service
            model_service = LangChainModelService()
            
            # Store in app state
            app.state.langchain_model_service = model_service
            app.state.langchain_service = LangChainAgentService(db, model_service)
            
            # Log success
            logger.info("LangChain integration initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LangChain integration: {str(e)}")
        finally:
            db.close()
    
    # Add LangChain execution to existing agent service
    from agent_service import AgentService
    
    # Patch the AgentService.execute_agent method to try LangChain first
    original_execute = AgentService.execute_agent
    
    async def execute_with_langchain(self, agent_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with LangChain if possible, fall back to regular execution"""
        # Get the agent from database
        db_agent = self.db.query(AgentDB).filter(AgentDB.id == agent_id).first()
        if not db_agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID {agent_id} not found"
            )
        
        if not db_agent.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Agent {db_agent.name} is inactive"
            )
        
        # Check if this is a LangChain agent
        try:
            # Try to parse code as JSON to identify LangChain agents
            json.loads(db_agent.code)
            
            # This appears to be a LangChain agent
            if hasattr(app.state, "langchain_service"):
                langchain_service = app.state.langchain_service
                
                # Get agent config or create it
                agent_config = langchain_service.loaded_agents.get(agent_id)
                if not agent_config:
                    agent_data = json.loads(db_agent.code)
                    agent_config = await langchain_service.create_agent_from_specifications(agent_data)
                    langchain_service.loaded_agents[agent_id] = agent_config
                
                # Execute the agent
                start_time = time.time()
                result = await langchain_service.execute_agent(agent_config, input_data)
                execution_time = (time.time() - start_time) * 1000
                
                # Update statistics
                self._update_execution_stats(db_agent, execution_time)
                
                return {
                    "output": result,
                    "execution_time_ms": execution_time,
                    "agent_id": agent_id,
                    "agent_name": db_agent.name,
                    "agent_version": db_agent.version
                }
        except (json.JSONDecodeError, Exception) as e:
            # Not a LangChain agent or error, fall back to regular execution
            logger.info(f"Falling back to regular execution for agent {agent_id}: {str(e)}")
        
        # If not a LangChain agent or if LangChain execution failed, use regular execution
        return await original_execute(self, agent_id, input_data)
    
    # Replace the method
    AgentService.execute_agent = execute_with_langchain