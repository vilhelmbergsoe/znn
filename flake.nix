{
  description = "znn - Tensor library in Zig";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    zig.url = "github:mitchellh/zig-overlay";

    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    ...
  } @ inputs: let
    overlays = [
      (final: prev: rec {
        zigpkgs = inputs.zig.packages.${prev.system};
        zls = inputs.nixpkgs.legacyPackages.${prev.system}.zls.overrideAttrs(prev: {
          nativeBuildInputs = [ zigpkgs.master];
        });
      })
    ];

    systems = builtins.attrNames inputs.zig.packages;
  in
    flake-utils.lib.eachSystem systems (
      system: let
        pkgs = import nixpkgs {inherit overlays system;};
      in rec {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            zigpkgs.master
            # zls
          ];
        };

        devShell = self.devShells.${system}.default;
      }
    );
}
