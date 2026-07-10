class DSGuardianError(Exception):
    """Clase base de excepción para el framework DS Guardian."""
    pass

class DataValidationError(DSGuardianError):
    """Excepción lanzada cuando los datos no pasan la validación de sanidad o auditoría."""
    pass

class ModelAuditingError(DSGuardianError):
    """Excepción lanzada cuando hay un error en la evaluación u optimización del modelo."""
    pass
