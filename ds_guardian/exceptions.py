class DSGuardianError(Exception):
    """Excepción base para todos los errores de DS Guardian."""
    pass

class DataValidationError(DSGuardianError):
    """Excepción lanzada cuando los datos de entrada son inválidos, están vacíos o no cumplen con los tipos esperados."""
    pass

class DataLeakageError(DSGuardianError):
    """Excepción de alta severidad lanzada cuando se detecta fuga de datos (Data Leakage) entre Train/Test o Target."""
    pass

class ImbalanceWarning(UserWarning):
    """Advertencia de desbalance severo de clases en la variable objetivo."""
    pass
