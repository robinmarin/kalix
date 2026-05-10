/// Expression AST nodes for symbolic dynamics.
///
/// Expressions are polynomial in state variables and the special token `dt`.
/// Supported operations: addition, subtraction, multiplication, division-by-literal,
/// and integer powers.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A numeric literal: `3.14`, `0.5`, `1.0`
    Lit(f64),
    /// A variable reference: state variable name (e.g. `"pos"`) or `"dt"`
    Var(String),
    /// Addition: `a + b`
    Add(Box<Expr>, Box<Expr>),
    /// Subtraction: `a - b`
    Sub(Box<Expr>, Box<Expr>),
    /// Multiplication: `a * b`
    Mul(Box<Expr>, Box<Expr>),
    /// Division: `a / b` — right-hand side must be a literal
    Div(Box<Expr>, Box<Expr>),
    /// Integer power: `a ^ n` where n is a positive integer
    Pow(Box<Expr>, u32),
}

impl Expr {
    /// Create a `Lit` node.
    pub fn lit(v: f64) -> Self {
        Expr::Lit(v)
    }

    /// Create a `Var` node.
    pub fn var(name: &str) -> Self {
        Expr::Var(name.to_string())
    }

    /// Convenience: `a + b`
    #[allow(clippy::should_implement_trait)]
    pub fn add(a: Expr, b: Expr) -> Self {
        Expr::Add(Box::new(a), Box::new(b))
    }

    /// Convenience: `a - b`
    #[allow(clippy::should_implement_trait)]
    pub fn sub(a: Expr, b: Expr) -> Self {
        Expr::Sub(Box::new(a), Box::new(b))
    }

    /// Convenience: `a * b`
    #[allow(clippy::should_implement_trait)]
    pub fn mul(a: Expr, b: Expr) -> Self {
        Expr::Mul(Box::new(a), Box::new(b))
    }

    /// Convenience: `a / b`
    #[allow(clippy::should_implement_trait)]
    pub fn div(a: Expr, b: Expr) -> Self {
        Expr::Div(Box::new(a), Box::new(b))
    }

    /// Convenience: `a ^ n`
    pub fn pow(a: Expr, n: u32) -> Self {
        Expr::Pow(Box::new(a), n)
    }
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Lit(v) => write!(f, "{}", v),
            Expr::Var(name) => write!(f, "{}", name),
            Expr::Add(a, b) => write!(f, "({} + {})", a, b),
            Expr::Sub(a, b) => write!(f, "({} - {})", a, b),
            Expr::Mul(a, b) => write!(f, "({} * {})", a, b),
            Expr::Div(a, b) => write!(f, "({} / {})", a, b),
            Expr::Pow(a, n) => write!(f, "({} ^ {})", a, n),
        }
    }
}
