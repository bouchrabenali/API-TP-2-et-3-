import uvicorn
from src.app import get_application
from src.services.utils import setup_logger
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from src.services.rate_limiter import limiter, _rate_limit_exceeded_handler, RateLimitExceeded
app = get_application()
logger = setup_logger()


app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)



@app.get("/", include_in_schema=False)
def root():
    return responses.RedirectResponse(url='/docs')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
