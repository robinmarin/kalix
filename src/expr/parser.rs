//! String → AST parser for symbolic dynamics expressions.
//!
//! Grammar (precedence low to high):
//! ```text
//! expr   → term (('+' | '-') term)*
//! term   → power (('*' | '/') power)*
//! power  → atom ('^' u32)?
//! atom   → '(' expr ')' | NUMBER | IDENT
//! ```

use super::ast::Expr;
use std::iter::Peekable;
use std::str::Chars;

/// Parse a dynamics expression string into an AST.
///
/// # Errors
///
/// Returns a descriptive error string for unsupported syntax (trig functions,
/// division by variables, etc.).
pub fn parse(input: &str) -> Result<Expr, String> {
    let mut lexer = Lexer::new(input);
    let expr = parse_expr(&mut lexer)?;
    if let Some(tok) = lexer.peek() {
        return Err(format!("unexpected token after expression: {:?}", tok));
    }
    // Post-validate: division right-hand side must be a literal
    validate_division(&expr)?;
    Ok(expr)
}

/// Validate that every `Div` node has a `Lit` on the right-hand side.
fn validate_division(expr: &Expr) -> Result<(), String> {
    match expr {
        Expr::Lit(_) | Expr::Var(_) => Ok(()),
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) => {
            validate_division(a)?;
            validate_division(b)?;
            Ok(())
        }
        Expr::Div(_, b) => {
            if !matches!(b.as_ref(), Expr::Lit(_)) {
                return Err(
                    "division by variable disallowed — right-hand side must be a literal"
                        .to_string(),
                );
            }
            validate_division(b)?;
            Ok(())
        }
        Expr::Pow(a, _) => validate_division(a),
        Expr::Sin(a) | Expr::Cos(a) => validate_division(a),
    }
}

// ── Tokeniser ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
}

struct Lexer<'a> {
    chars: Peekable<Chars<'a>>,
    peeked: Option<Token>,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Lexer {
            chars: input.chars().peekable(),
            peeked: None,
        }
    }

    fn peek(&mut self) -> Option<Token> {
        if self.peeked.is_none() {
            self.peeked = self.next_token();
        }
        self.peeked.clone()
    }

    fn next(&mut self) -> Option<Token> {
        if self.peeked.is_some() {
            self.peeked.take()
        } else {
            self.next_token()
        }
    }

    fn next_token(&mut self) -> Option<Token> {
        // Skip whitespace
        while let Some(&ch) = self.chars.peek() {
            if ch.is_whitespace() {
                self.chars.next();
            } else {
                break;
            }
        }

        match self.chars.next() {
            None => None,
            Some(ch) => match ch {
                '+' => Some(Token::Plus),
                '-' => Some(Token::Minus),
                '*' => Some(Token::Star),
                '/' => Some(Token::Slash),
                '^' => Some(Token::Caret),
                '(' => Some(Token::LParen),
                ')' => Some(Token::RParen),
                c if c.is_ascii_digit() || c == '.' => {
                    let mut num = String::from(c);
                    while let Some(&nc) = self.chars.peek() {
                        if nc.is_ascii_digit() || nc == '.' {
                            num.push(self.chars.next().unwrap());
                        } else {
                            break;
                        }
                    }
                    let value: f64 = num
                        .parse()
                        .map_err(|_| format!("invalid number: {}", num))
                        .ok()?;
                    Some(Token::Number(value))
                }
                c if c.is_alphabetic() || c == '_' => {
                    let mut ident = String::from(c);
                    while let Some(&nc) = self.chars.peek() {
                        if nc.is_alphanumeric() || nc == '_' {
                            ident.push(self.chars.next().unwrap());
                        } else {
                            break;
                        }
                    }
                    Some(Token::Ident(ident))
                }
                other => {
                    // Store error and return None — caller will detect missing token
                    eprintln!("unexpected character: '{}'", other);
                    None
                }
            },
        }
    }
}

// ── Recursive Descent Parser ───────────────────────────────────────────

fn parse_expr(lexer: &mut Lexer) -> Result<Expr, String> {
    let mut left = parse_term(lexer)?;
    loop {
        match lexer.peek() {
            Some(Token::Plus) => {
                lexer.next();
                let right = parse_term(lexer)?;
                left = Expr::add(left, right);
            }
            Some(Token::Minus) => {
                lexer.next();
                let right = parse_term(lexer)?;
                left = Expr::sub(left, right);
            }
            _ => break,
        }
    }
    Ok(left)
}

fn parse_term(lexer: &mut Lexer) -> Result<Expr, String> {
    let mut left = parse_power(lexer)?;
    loop {
        match lexer.peek() {
            Some(Token::Star) => {
                lexer.next();
                let right = parse_power(lexer)?;
                left = Expr::mul(left, right);
            }
            Some(Token::Slash) => {
                lexer.next();
                let right = parse_power(lexer)?;
                left = Expr::div(left, right);
            }
            _ => break,
        }
    }
    Ok(left)
}

fn parse_power(lexer: &mut Lexer) -> Result<Expr, String> {
    let base = parse_atom(lexer)?;
    match lexer.peek() {
        Some(Token::Caret) => {
            lexer.next();
            // Next must be an integer literal
            match lexer.next() {
                Some(Token::Number(n)) => {
                    if n.fract() != 0.0 || n < 0.0 {
                        return Err(format!(
                            "power exponent must be a positive integer, got {}",
                            n
                        ));
                    }
                    Ok(Expr::pow(base, n as u32))
                }
                other => Err(format!(
                    "expected integer exponent after '^', got {:?}",
                    other
                )),
            }
        }
        _ => Ok(base),
    }
}

fn parse_atom(lexer: &mut Lexer) -> Result<Expr, String> {
    match lexer.next() {
        Some(Token::Number(v)) => Ok(Expr::Lit(v)),
        Some(Token::Ident(name)) => {
            // Function call: sin(...), cos(...)
            if let Some(Token::LParen) = lexer.peek() {
                lexer.next(); // consume '('
                let arg = parse_expr(lexer)?;
                match lexer.next() {
                    Some(Token::RParen) => match name.as_str() {
                        "sin" => Ok(Expr::sin(arg)),
                        "cos" => Ok(Expr::cos(arg)),
                        other => Err(format!(
                            "unknown function: '{}' — only sin and cos are supported",
                            other
                        )),
                    },
                    other => Err(format!("expected ')', got {:?}", other)),
                }
            } else {
                Ok(Expr::Var(name))
            }
        }
        Some(Token::LParen) => {
            let expr = parse_expr(lexer)?;
            match lexer.next() {
                Some(Token::RParen) => Ok(expr),
                other => Err(format!("expected ')', got {:?}", other)),
            }
        }
        Some(other) => Err(format!("unexpected token: {:?}", other)),
        None => Err("unexpected end of input".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal() {
        assert_eq!(parse("3.14"), Ok(Expr::Lit(3.14)));
    }

    #[test]
    fn test_variable() {
        assert_eq!(parse("vel"), Ok(Expr::Var("vel".to_string())));
    }

    #[test]
    fn test_addition() {
        assert_eq!(
            parse("pos + vel"),
            Ok(Expr::add(Expr::var("pos"), Expr::var("vel")))
        );
    }

    #[test]
    fn test_multiplication() {
        assert_eq!(
            parse("vel * dt"),
            Ok(Expr::mul(Expr::var("vel"), Expr::var("dt")))
        );
    }

    #[test]
    fn test_power() {
        assert_eq!(parse("dt^2"), Ok(Expr::pow(Expr::var("dt"), 2)));
    }

    #[test]
    fn test_complex_expression() {
        let result = parse("pos + vel*dt + 0.5*acc*dt^2");
        assert!(result.is_ok(), "Expected OK, got {:?}", result);
    }

    #[test]
    fn test_sin_parses() {
        let result = parse("sin(pos)");
        assert!(
            result.is_ok(),
            "Expected sin(pos) to parse, got {:?}",
            result
        );
        assert_eq!(result.unwrap(), Expr::sin(Expr::var("pos")));
    }

    #[test]
    fn test_cos_parses() {
        let result = parse("cos(vel)");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Expr::cos(Expr::var("vel")));
    }

    #[test]
    fn test_sin_with_expression() {
        let result = parse("sin(vel*dt)");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            Expr::sin(Expr::mul(Expr::var("vel"), Expr::var("dt"))),
        );
    }

    #[test]
    fn test_unknown_function_rejected() {
        assert!(parse("tan(pos)").is_err());
    }

    #[test]
    fn test_division_by_variable_disallowed() {
        let result = parse("pos / vel");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("division by variable"));
    }

    #[test]
    fn test_division_by_literal_allowed() {
        let result = parse("pos / 2.0");
        assert!(result.is_ok());
    }

    #[test]
    fn test_precedence() {
        // a + b * c  →  a + (b * c)
        let result = parse("a + b * c");
        assert!(result.is_ok());
        if let Ok(Expr::Add(_, right)) = result {
            assert!(matches!(*right, Expr::Mul(_, _)));
        } else {
            panic!("expected Add node");
        }
    }
}
