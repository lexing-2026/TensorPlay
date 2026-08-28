# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Use GitHub's private vulnerability reporting feature for this repository
(Security tab → "Report a vulnerability"). Reports are visible only to the
repository maintainers.

You should receive an initial response within 7 days. If the report is
accepted, we aim to publish a fix and a security advisory within 90 days,
coordinating disclosure timing with you.

Please include:

- A description of the issue and its potential impact.
- Steps or a minimal proof of concept to reproduce it.
- The affected version(s) and install source (PyPI, CUDA index, source build).

## Scope

TensorPlay is an educational deep learning framework. Security issues in
third-party dependencies under `third_party/` (pytorch, tilelang, ...) should
be reported to the respective upstream projects instead.
