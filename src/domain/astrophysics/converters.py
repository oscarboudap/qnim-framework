import numpy as np

def extract_physical_params(m_chirp_det, eta=0.245):
    """
    Convierte la Masa de Chirp detectada en masas individuales (m1, m2).
    Basado en el formalismo de estimación de parámetros LVK.
    """
    # m_total = M_chirp / eta^(3/5)
    m_total = m_chirp_det / (eta**(3/5))
    
    # Resolviendo m1, m2 a partir de la masa total y el ratio simétrico eta
    # m1,2 = (M/2) * (1 +/- sqrt(1 - 4*eta))
    diff = np.sqrt(max(0, 1 - 4 * eta))
    m1 = (m_total / 2) * (1 + diff)
    m2 = (m_total / 2) * (1 - diff)
    
    return m1, m2