resource "aws_db_instance" "vector_storage" {
  allocated_storage    = 20
  engine               = "mysql"
  engine_version       = "8.0"
  instance_class       = "db.t2.micro"
  name                 = "vectorstorage"
  username             = "admin"
  password             = "password"
  parameter_group_name = "default.mysql8.0"
  publicly_accessible  = false
  vpc_security_group_ids = [aws_security_group.default.id]
  db_subnet_group_name = aws_db_subnet_group.default.name
}

resource "aws_db_subnet_group" "default" {
  name       = "main"
  subnet_ids = [aws_subnet.subnet1.id, aws_subnet.subnet2.id]
}

resource "aws_security_group" "default" {
  vpc_id = aws_vpc.main.id

  ingress {
    from_port   = 3306
    to_port     = 3306
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
