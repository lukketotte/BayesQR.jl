push!(LOAD_PATH, "../src/")

using Documenter, BayesQR

makedocs(
    sitename = "BayesQR.jl",
    modules = [BayesQR],
    pages = [
    "Home" => "index.md"
    ])

deploydocs(;repo="github.com/lukketotte/BayesQR.jl")
