variable "project_name" {
  type    = string
  default = "data-at-scale"
}

variable "account_number" {
  type = string
}

variable "account_alias" {
  type    = string
  default = "cdca"
}

variable "region" {
  type = string
}

variable "env" {
  type = string
}

variable "env_category" {
  type = string
}

variable "environment" {
  type = string
}

variable "env_type" {
  type = string
}

variable "application" {
  type = string
}

variable "tags" {
  description = "A mapping of tags to assign to the resource"
  type        = map(string)
}

variable "app_version" {
  type = string
}
