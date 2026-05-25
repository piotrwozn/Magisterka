package com.sorai.gateway.client;

import com.sorai.gateway.dto.HealthStatus;
import com.sorai.gateway.dto.PredictRequest;
import com.sorai.gateway.dto.PredictResponse;
import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import io.github.resilience4j.retry.annotation.Retry;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

/** Calls the FastAPI ML service. */
@Component
public class MlServiceClient {

    private final WebClient mlClient;

    @Autowired
    public MlServiceClient(WebClient mlClient) {
        this.mlClient = mlClient;
    }

    @CircuitBreaker(name = "mlService", fallbackMethod = "predictFallback")
    @Retry(name = "mlService")
    public Mono<PredictResponse> predict(PredictRequest request) {
        return mlClient.post()
                .uri("/api/v1/predict")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(request)
                .retrieve()
                .bodyToMono(PredictResponse.class);
    }

    public Mono<HealthStatus> health() {
        return mlClient.get()
                .uri("/api/v1/health")
                .retrieve()
                .bodyToMono(HealthStatus.class);
    }

    /** Fallback when the ML service is unreachable. */
    @SuppressWarnings("unused")
    private Mono<PredictResponse> predictFallback(PredictRequest req, Throwable t) {
        return Mono.error(new MlServiceUnavailableException(
                "ML inference service unreachable: " + t.getMessage(), t));
    }

    public static class MlServiceUnavailableException extends RuntimeException {
        public MlServiceUnavailableException(String msg, Throwable cause) {
            super(msg, cause);
        }
    }
}
