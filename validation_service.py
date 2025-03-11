from typing import Dict, List, Any, Optional
import json
import os
import asyncio
from datetime import datetime
import logging
import time
import uuid
import re
import importlib

# FastAPI imports
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger("validation_service")

# Custom validation service
class ValidationRule:
    @staticmethod
    def validate_type(value: Any, expected_type: str) -> Dict[str, Any]:
        """Validate data type"""
        if expected_type == "string" and not isinstance(value, str):
            return {"valid": False, "message": f"Value must be a string"}
        elif expected_type == "number" and not isinstance(value, (int, float)):
            return {"valid": False, "message": f"Value must be a number"}
        elif expected_type == "boolean" and not isinstance(value, bool):
            return {"valid": False, "message": f"Value must be a boolean"}
        elif expected_type == "array" and not isinstance(value, list):
            return {"valid": False, "message": f"Value must be an array"}
        elif expected_type == "object" and not isinstance(value, dict):
            return {"valid": False, "message": f"Value must be an object"}
        return {"valid": True, "message": f"Value is a valid {expected_type}"}
    
    @staticmethod
    def validate_range(value: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Dict[str, Any]:
        """Validate numeric range"""
        if not isinstance(value, (int, float)):
            return {"valid": False, "message": f"Value must be a number for range validation"}
        
        if min_val is not None and value < min_val:
            return {"valid": False, "message": f"Value must be at least {min_val}"}
        
        if max_val is not None and value > max_val:
            return {"valid": False, "message": f"Value must be at most {max_val}"}
        
        return {"valid": True, "message": f"Value is within valid range"}
    
    @staticmethod
    def validate_regex(value: str, pattern: str) -> Dict[str, Any]:
        """Validate against regex pattern"""
        if not isinstance(value, str):
            return {"valid": False, "message": f"Value must be a string for regex validation"}
        
        if re.match(pattern, value):
            return {"valid": True, "message": f"Value matches the required pattern"}
        else:
            return {"valid": False, "message": f"Value must match pattern {pattern}"}
    
    @staticmethod
    def validate_enum(value: Any, allowed_values: List[Any]) -> Dict[str, Any]:
        """Validate against enum values"""
        if value in allowed_values:
            return {"valid": True, "message": f"Value is in the allowed list"}
        else:
            return {"valid": False, "message": f"Value must be one of {allowed_values}"}

class ValidationService:
    """A custom validation service"""
    
    def __init__(self):
        self.validators = {
            "type": ValidationRule.validate_type,
            "range": ValidationRule.validate_range,
            "regex": ValidationRule.validate_regex,
            "enum": ValidationRule.validate_enum
        }
    
    def validate_input(self, input_data: Dict[str, Any], validations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate input data against validation rules"""
        validation_results = []
        all_valid = True
        
        for validation in validations:
            field = validation.get("field", "")
            rule_type = validation.get("rule_type", "")
            parameters = validation.get("parameters", {})
            error_message = validation.get("error_message")
            
            # Skip validation if field is not in input data
            if field not in input_data:
                continue
            
            # Get the validator function
            validator = self.validators.get(rule_type)
            if not validator:
                continue
            
            # Apply validation
            if rule_type == "type":
                result = validator(input_data[field], parameters.get("type", "string"))
            elif rule_type == "range":
                result = validator(input_data[field], parameters.get("min"), parameters.get("max"))
            elif rule_type == "regex":
                result = validator(input_data[field], parameters.get("pattern", ""))
            elif rule_type == "enum":
                result = validator(input_data[field], parameters.get("values", []))
            else:
                continue
            
            # If validation failed, add to results
            if not result["valid"]:
                all_valid = False
                validation_results.append({
                    "field": field,
                    "valid": False,
                    "message": error_message or result["message"]
                })
        
        return {
            "valid": all_valid,
            "validation_results": validation_results
        }

# Pydantic models for API
class EnhancedAgentExecutionRequest(BaseModel):
    input_data: Dict[str, Any]

class EnhancedAgentExecutionResponse(BaseModel):
    success: bool
    output: Optional[Dict[str, Any]] = None
    validation_errors: Optional[List[Dict[str, Any]]] = None
    execution_time_ms: Optional[float] = None
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    agent_version: Optional[int] = None
    message: Optional[str] = None

# FastAPI router
validation_router = APIRouter(prefix="/enhanced", tags=["Enhanced Validation"])

# This will be set up when the module is imported
db_dependency = None
AgentDB = None

# Function to execute agent with validation
async def execute_with_validation(agent_id: str, input_data: Dict[str, Any], db):
    """Execute an agent with validation"""
    # We need to import these here to avoid circular imports
    global AgentDB
    if AgentDB is None:
        # Dynamically import from agent_service
        import agent_service as agent_service_module
        AgentDB = agent_service_module.AgentDB
    
    validation_service = ValidationService()
    
    # Get agent from database
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
    
    # Validate input
    validation_result = validation_service.validate_input(input_data, db_agent.validations)
    if not validation_result["valid"]:
        return {
            "success": False,
            "validation_errors": validation_result["validation_results"],
            "message": "Input validation failed"
        }
    
    # Import the agent_service module to use its functions
    import agent_service as agent_service_module
    
    # Get a new instance of AgentService
    agent_service = agent_service_module.AgentService(
        db, 
        agent_service_module.GeminiService(), 
        agent_service_module.PromptGeneratorService()
    )
    
    # Load agent module
    try:
        agent_module = await agent_service._load_agent_module(agent_id, db_agent.name, db_agent.code)
    except Exception as e:
        logger.error(f"Error loading agent module: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load agent code: {str(e)}"
        )
    
    # Execute agent
    start_time = time.time()
    try:
        # Run in threadpool
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            agent_module.process,
            input_data
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        # Update stats
        agent_service._update_execution_stats(db_agent, execution_time)
        
        return {
            "success": True,
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

# This function will be called by agent_service.py
def setup_routes(app, get_db_func):
    """Set up the routes for the validation service"""
    global db_dependency
    db_dependency = get_db_func
    
    # Define the endpoint
    @validation_router.post("/agents/{agent_id}/execute", response_model=EnhancedAgentExecutionResponse)
    async def enhanced_execute_endpoint(
        agent_id: str,
        request: EnhancedAgentExecutionRequest,
        db = Depends(db_dependency)
    ):
        """Execute an agent with enhanced validation"""
        return await execute_with_validation(agent_id, request.input_data, db)
    
    # Include router in app
    app.include_router(validation_router)
    
    # Log success
    logger.info("Enhanced validation routes set up successfully")