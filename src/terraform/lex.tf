resource "aws_lex_bot" "cybersecurity_bot" {
  name = "CybersecurityBot"
  description = "A bot for handling cybersecurity inquiries and alert analysis"
  locale = "en-US"
  child_directed = false

  intent {
    intent_name = aws_lex_intent.explain_edr_alert.name
    intent_version = aws_lex_intent.explain_edr_alert.version
  }

  voice_id = "Joanna"
}

resource "aws_lex_intent" "explain_edr_alert" {
  name = "ExplainEDRAlert"
  description = "Intent to explain EDR alerts"

  sample_utterances = [
    "Explain this EDR alert: {alert_details}",
    "What does this alert mean: {alert_details}"
  ]

  slot {
    name = "alert_details"
    slot_type = "AMAZON.LITERAL"
    slot_constraint = "Required"
    value_elicitation_prompt {
      messages {
        content_type = "PlainText"
        content = "Please provide the details of the EDR alert."
      }
      max_attempts = 3
    }
  }

  fulfillment_activity {
    type = "CodeHook"
    code_hook {
      message_version = "1.0"
      uri = aws_lambda_function.explain_edr_alert.invoke_arn
    }
  }
}
