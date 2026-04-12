import os
import sys
from dotenv import load_dotenv

# Aseguramos que Python detecte el módulo 'src' desde la raíz
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.application.model_training_service import ModelTrainingService

def main():
    load_dotenv()
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Instanciamos y ejecutamos el servicio de la capa de aplicación
    trainer_service = ModelTrainingService(project_root=project_root)
    
    # Entrenamos con 20 eventos (puedes subir este número luego)
    trainer_service.execute_training(max_events=20)

if __name__ == "__main__":
    main()