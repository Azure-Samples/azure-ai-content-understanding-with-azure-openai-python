# Clear the contents of the .env file
Set-Content -Path notebooks/.env -Value ""

# Append new values to the .env file
$v1 = azd env get-value AZURE_AI_SERVICE_ENDPOINT
$v2 = azd env get-value AZURE_OPENAI_ENDPOINT
$v3 = azd env get-value AZURE_OPENAI_CHAT_DEPLOYMENT_NAME

Add-Content -Path notebooks/.env -Value "AZURE_AI_SERVICE_ENDPOINT=$v1"
Add-Content -Path notebooks/.env -Value "AZURE_OPENAI_ENDPOINT=$v2"
Add-Content -Path notebooks/.env -Value "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=$v3"