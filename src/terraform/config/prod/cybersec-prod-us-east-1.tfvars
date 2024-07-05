# Project_name is simply used a naming prefix for several resources.
project_name   = "cybersec-bot"
env            = "prod"
env_category   = "prod"
region         = "us-east-1"
environment    = "prod"
application    = "cybersec-bot"
account_number = "15647283869" # whatever it actually is, replace this

env_type = "prod"
cybersec_bot_folder = "cybersec-bot/prod"

# it's good practice to have a tagging policy
tags = {
  project      = "cybersec-bot"
  owner        = "victor.gueorg@gmail.com"
  env          = "prod"
  env_category = "prod"
  email        = "victor.gueorg@gmail.com"
  sguid        = "product:platform:cybersec"
  src_url      = "https://github.com/BrutishGuy/LLM4CyberSec"
}