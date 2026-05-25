package com.sorai.gateway.error;

import com.sorai.gateway.client.MlServiceClient.MlServiceUnavailableException;
import io.github.resilience4j.circuitbreaker.CallNotPermittedException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.reactive.function.client.WebClientResponseException;

import java.time.Instant;
import java.util.Map;
import java.util.stream.Collectors;

@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(MlServiceUnavailableException.class)
    public ResponseEntity<Map<String, Object>> mlDown(MlServiceUnavailableException ex) {
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                .body(errorBody(503, "ML inference service unavailable", ex.getMessage()));
    }

    @ExceptionHandler(CallNotPermittedException.class)
    public ResponseEntity<Map<String, Object>> circuitOpen(CallNotPermittedException ex) {
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                .body(errorBody(503, "Circuit breaker open — ML service unhealthy", ex.getMessage()));
    }

    @ExceptionHandler(WebClientResponseException.class)
    public ResponseEntity<Map<String, Object>> mlError(WebClientResponseException ex) {
        return ResponseEntity.status(ex.getStatusCode())
                .body(errorBody(ex.getStatusCode().value(), "ML service error", ex.getResponseBodyAsString()));
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<Map<String, Object>> validation(MethodArgumentNotValidException ex) {
        String details = ex.getBindingResult().getFieldErrors().stream()
                .map(f -> f.getField() + ": " + f.getDefaultMessage())
                .collect(Collectors.joining("; "));
        return ResponseEntity.status(HttpStatus.UNPROCESSABLE_ENTITY)
                .body(errorBody(422, "Validation failed", details));
    }

    @ExceptionHandler(Throwable.class)
    public ResponseEntity<Map<String, Object>> generic(Throwable ex) {
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(errorBody(500, "Internal error", ex.getMessage()));
    }

    private static Map<String, Object> errorBody(int status, String error, String detail) {
        return Map.of(
                "timestamp", Instant.now().toString(),
                "status",    status,
                "error",     error,
                "detail",    detail == null ? "" : detail
        );
    }
}
