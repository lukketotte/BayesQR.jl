push!(LOAD_PATH, "../src/")

using Documenter
using BayesQR

makedocs(
    sitename = "BayesQR.jl",
    modules = [BayesQR],
    pages = [
    "Home" => "index.md"
    ])

deploydocs(;repo="github.com/lukketotte/BayesQr.jl")
