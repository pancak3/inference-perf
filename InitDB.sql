CREATE SCHEMA IF NOT EXISTS metrics;
CREATE TABLE metrics."inference-instance" (
    "RequestID"            TEXT PRIMARY KEY,
    "StartGenerationAt"    BIGINT NOT NULL,
    "RespondFirstTokenAt"  BIGINT NOT NULL,
    "RespondLastTokenAt"   BIGINT NOT NULL,
    "AggregatedQueryHits"  INT NOT NULL,
    "GeneratedTokens"   BIGINT NOT NULL,
    "PromptTokens"   BIGINT NOT NULL
);
CREATE TABLE metrics."client" (
    "RequestID"            TEXT PRIMARY KEY,
    "SendAt"    BIGINT NOT NULL,
    "ReceiveFirstTokenAt"  BIGINT NOT NULL,
    "ReceiveLastTokenAt"   BIGINT NOT NULL,
    "MaxCompletionTokens"   BIGINT NOT NULL,
    "GeneratedTokens"   BIGINT NOT NULL,
    "ResponseStatus" INTEGER NOT NULL
);

CREATE TABLE metrics."scheduler" (
    "RequestID"            TEXT PRIMARY KEY,
    "ReceiveRequestAt"    BIGINT NOT NULL,
    "SchedulerStartAt"  BIGINT NOT NULL,
    "ServiceSelectedAt"   BIGINT NOT NULL,
    "SelectedActor" TEXT NOT NULL
);

CREATE TABLE metrics."node" (
    "NodeName"    TEXT NOT NULL,
    "Timestamp"    BIGINT NOT NULL,
    "CPUUtilization"   FLOAT NOT NULL,
    "MemoryUtilization"   FLOAT NOT NULL,
    "GPUUtilization"  FLOAT NOT NULL,
    "NetWorkIn"   BIGINT NOT NULL,
    "NetWorkOut"   BIGINT NOT NULL,
    "DiskRead"   BIGINT NOT NULL,
    "DiskWrite"   BIGINT NOT NULL
);

CREATE VIEW metrics.requests AS
SELECT 
    ii."RequestID",
    c."SendAt",
    s."ReceiveRequestAt",
    s."SchedulerStartAt",
    s."ServiceSelectedAt",
    s."SelectedActor",
    ii."StartGenerationAt",
    ii."RespondFirstTokenAt",
    c."ReceiveFirstTokenAt",
    ii."RespondLastTokenAt",
    c."ReceiveLastTokenAt",
    ii."AggregatedQueryHits",
    ii."PromptTokens",
    c."MaxCompletionTokens" as "ClientMaxCompletionTokens",
    c."GeneratedTokens" as "ClientGeneratedTokens",
    ii."GeneratedTokens" as "ServerGeneratedTokens",
    c."ResponseStatus"

FROM 
    metrics."inference-instance" ii
INNER JOIN 
    metrics."client" c ON ii."RequestID" = c."RequestID"
INNER JOIN 
    metrics."scheduler" s ON ii."RequestID" = s."RequestID";

CREATE VIEW metrics.request_node_metrics AS
SELECT * FROM metrics."node"
WHERE "Timestamp" BETWEEN (SELECT MIN("SendAt") FROM metrics.requests) AND (SELECT MAX("ReceiveLastTokenAt") FROM metrics.requests);
