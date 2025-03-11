from fastapi import FastAPI, Depends
from typing import List, Dict, Any
import uvicorn
import json
from sqlalchemy.orm import Session

# Import the necessary components from agent_service
from agent_service import get_db, AgentDB

app = FastAPI()

@app.get("/langchain/agents", response_model=List[Dict[str, Any]])
async def list_langchain_agents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
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

# Run this server on a different port
if __name__ == "__main__":
    uvicorn.run("fixed_langchain_api:app", host="0.0.0.0", port=8001)