# FAIVOR-ML-Validator Error Handling Improvements

## Summary of Changes

This document outlines the error handling improvements made to the FAIVOR-ML-Validator service to provide better, more structured error information to clients.

### 1. API Controller Enhancements (`api_controller.py`)

#### Added Structured Error Models
- Created `ErrorDetail` and `ErrorResponse` Pydantic models for consistent error formatting
- Added `create_error_response` helper function to generate structured errors

#### Error Response Format
```json
{
  "detail": {
    "code": "ERROR_CODE",
    "message": "User-friendly error message",
    "technical_details": "Technical error details for debugging",
    "metadata": {
      "additional": "context"
    }
  }
}
```

#### Endpoint Updates
- **`/validate-csv/`**:
  - Returns structured errors for invalid metadata JSON, CSV parsing errors
  - Includes missing columns information in validation response
  - Uses appropriate HTTP status codes (400 for client errors)

- **`/retrieve-metrics`**:
  - Returns structured errors for JSON parsing and metric retrieval failures

- **`/validate-model`**:
  - Enhanced error handling for each stage of validation
  - Specific error codes for different failure types:
    - `CONTAINER_EXECUTION_ERROR`: Docker container issues
    - `MODEL_EXECUTION_TIMEOUT`: Execution timeouts
    - `METRICS_CALCULATION_ERROR`: Metric computation failures
  - Returns 503 for service issues, 400 for client errors, 500 for server errors

### 2. Docker Execution Improvements (`run_docker.py`)

#### Enhanced Container Startup
- Better error messages for missing Docker daemon
- Specific errors for image not found
- Detailed container startup failure messages
- Includes container logs in error responses

#### Improved Error Context
- Timeout errors include actual elapsed time
- Container execution errors include recent logs
- Better error messages for model execution failures

#### Error Types Handled
- Docker daemon not available
- Image pull failures (not found, access denied)
- Container startup failures
- Execution timeouts with context
- Model prediction failures with logs

### 3. Metrics Calculation Improvements (`calculate_metrics.py`)

#### Data Validation
- Validates predictions and expected outputs are not empty
- Checks that arrays have matching lengths
- Provides clear error messages for missing data

#### Error Recovery
- Graceful handling of missing sensitive attributes
- Continues calculation even if some metric categories fail
- Returns partial results when possible

#### Better Error Messages
- Specific messages for data preparation failures
- Clear guidance on fixing numeric conversion issues
- Detailed error reporting for each metric category

## Benefits

1. **Better Debugging**: Technical details and error codes help identify issues quickly
2. **User-Friendly Messages**: Clear guidance on how to fix common problems
3. **Backward Compatible**: Maintains existing API contracts while adding new error details
4. **Consistent Format**: All errors follow the same structure across endpoints
5. **Appropriate Status Codes**: 400 for client errors, 503 for service issues, 500 for server errors

## Error Code Reference

- `INVALID_METADATA_JSON`: Malformed JSON in metadata
- `METADATA_PARSE_ERROR`: Valid JSON but invalid metadata structure
- `INVALID_CSV_FORMAT`: CSV parsing errors
- `CSV_READ_ERROR`: File reading issues
- `MISSING_REQUIRED_COLUMNS`: Required columns not found in CSV
- `CONTAINER_EXECUTION_ERROR`: Docker container failures
- `MODEL_EXECUTION_TIMEOUT`: Execution time exceeded
- `MODEL_EXECUTION_FAILED`: Model returned error status
- `METRICS_CALCULATION_ERROR`: Failed to compute metrics
- `METRICS_CALCULATOR_INIT_ERROR`: Failed to initialize calculator

## Testing Recommendations

1. Test with invalid JSON metadata
2. Test with missing Docker daemon
3. Test with non-existent Docker images
4. Test with timeout scenarios
5. Test with invalid CSV formats
6. Test with missing required columns
7. Verify frontend properly parses new error format