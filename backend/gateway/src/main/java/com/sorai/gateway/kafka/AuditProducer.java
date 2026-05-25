package com.sorai.gateway.kafka;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sorai.gateway.dto.PredictResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.util.Map;

/**
 * Asynchronously logs every prediction to Kafka for audit.
 *
 * IMPORTANT: clinical notes are NEVER logged — GDPR / RODO compliance.
 * Only outcome metadata is persisted.
 */
@Component
public class AuditProducer {

    private static final Logger log = LoggerFactory.getLogger(AuditProducer.class);

    private final KafkaTemplate<String, String> kafka;
    private final ObjectMapper mapper;

    @Value("${audit.topic}")
    private String topic;

    @Value("${audit.enabled:true}")
    private boolean enabled;

    public AuditProducer(KafkaTemplate<String, String> kafka, ObjectMapper mapper) {
        this.kafka  = kafka;
        this.mapper = mapper;
    }

    public void logPrediction(PredictResponse response, long durationMs, String clientIp) {
        if (!enabled) return;
        try {
            Map<String, Object> event = Map.of(
                    "timestamp",      Instant.now().toString(),
                    "finalCategory",  response.finalCategory(),
                    "confidence",     response.confidence(),
                    "alertDoctor",    response.conflict().alertDoctor(),
                    "conflict",       response.conflict().detected(),
                    "severity",       response.conflict().severity(),
                    "modelsUsed",     response.modelPredictions().size(),
                    "processingMs",   durationMs,
                    "clientIpHash",   Integer.toString(clientIp == null ? 0 : clientIp.hashCode())
            );
            String json = mapper.writeValueAsString(event);
            kafka.send(topic, json);
        } catch (JsonProcessingException e) {
            log.warn("Failed to serialise audit event: {}", e.getMessage());
        } catch (Exception e) {
            // Never let audit failure break the response path
            log.warn("Audit log failed (non-fatal): {}", e.getMessage());
        }
    }
}
