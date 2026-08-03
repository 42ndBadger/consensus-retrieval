{ pkgs ? import <nixpkgs> {} }:

let
  pythonEnv = pkgs.python3.withPackages(ps: with ps; [
    matplotlib
    pandas
  ]);
in
  pkgs.mkShell {
    buildInputs = with pkgs; [
      pythonEnv
      perf
    ];

    shellHook = ''
        echo hello
    '';
  }
