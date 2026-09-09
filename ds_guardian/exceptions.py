class DSGuardianError(Exception):
    """Excepción base para todos los errores de DS Guardian."""
    pass

class DataValidationError(DSGuardianError):
    """Excepción lanzada cuando los datos de entrada son inválidos o no cumplen los tipos esperados."""
    pass

class DataLeakageError(DSGuardianError):
    """Excepción lanzada cuando se detecta fuga de datos (Data Leakage)."""
    pass

class ModelAuditingError(DSGuardianError):
    """Excepción lanzada cuando la auditoría o validación del modelo falla."""
    pass

class ImbalanceWarning(UserWarning):
    """Advertencia de desbalance severo de clases."""
    pass
