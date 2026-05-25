package com.sorai.gateway.api;

import com.sorai.gateway.client.MlServiceClient;
import com.sorai.gateway.dto.HealthStatus;
import com.sorai.gateway.dto.PredictRequest;
import com.sorai.gateway.dto.PredictResponse;
import com.sorai.gateway.kafka.AuditProducer;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;

/** Public API gateway endpoints — frontend talks here, NOT to FastAPI directly. */
@RestController
@RequestMapping("/api/v1")
public class PredictController {

    private static final Logger log = LoggerFactory.getLogger(PredictController.class);

    private final MlServiceClient mlClient;
    private final AuditProducer audit;

    public PredictController(MlServiceClient mlClient, AuditProducer audit) {
        this.mlClient = mlClient;
        this.audit    = audit;
    }

    @PostMapping("/predict")
    public Mono<ResponseEntity<PredictResponse>> predict(
            @Valid @RequestBody PredictRequest request,
            ServerWebExchange exchange
    ) {
        long started = System.currentTimeMillis();
        String clientIp = exchange.getRequest().getRemoteAddress() == null
                ? "?"
                : exchange.getRequest().getRemoteAddress().getAddress().getHostAddress();

        return mlClient.predict(request)
                .doOnSuccess(resp -> {
                    long duration = System.currentTimeMillis() - started;
                    log.info("predict | final={} conf={} alert={} duration={}ms",
                            resp.finalCategory(), resp.confidence(),
                            resp.conflict().alertDoctor(), duration);
                    audit.logPrediction(resp, duration, clientIp);
                })
                .map(ResponseEntity::ok);
    }

    @GetMapping("/health")
    public Mono<ResponseEntity<HealthStatus>> health() {
        return mlClient.health().map(ResponseEntity::ok);
    }
}
