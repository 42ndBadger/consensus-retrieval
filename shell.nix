{ pkgs ? import <nixpkgs> {} }:

let
  pythonEnv = pkgs.python3.withPackages(ps: with ps; [
    matplotlib
    pandas
  ]);
in
  pkgs.mkShell {
    buildInputs = [
      pythonEnv
    ];

    shellHook = ''
      echo "Entering Python development shell with matplotlib and pandas."
      echo "To run the script, use: python scripts/tune_parameters.py"
    '';
  }
