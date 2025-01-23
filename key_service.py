from dotenv import load_dotenv


# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Autenticación para acceder a los modelos de OpenAI
openai.api_key = os.environ["OPENAI_API_KEY"]