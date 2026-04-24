"""Domain registry API using the refactored package layout."""

from plm_data.domains.base import (
    CoordinateSample,
    DomainFactory,
    DomainSpec,
    GmshDomainBuilder,
    GmshDomainSpec,
    build_gmsh_domain_model,
    build_gmsh_planar_domain_model,
    create_domain,
    get_domain_spec,
    get_gmsh_domain_dimension,
    is_gmsh_domain,
    is_gmsh_planar_domain,
    list_domain_specs,
    list_domains,
    register_domain,
    register_domain_spec,
    register_gmsh_domain,
    register_gmsh_planar_domain,
    sample_coordinate_region,
)

from plm_data import domains as _domains_package

_domains_package._load_domain_spec_modules()

__all__ = [
    "CoordinateSample",
    "DomainFactory",
    "DomainSpec",
    "GmshDomainBuilder",
    "GmshDomainSpec",
    "build_gmsh_domain_model",
    "build_gmsh_planar_domain_model",
    "create_domain",
    "get_domain_spec",
    "get_gmsh_domain_dimension",
    "is_gmsh_domain",
    "is_gmsh_planar_domain",
    "list_domain_specs",
    "list_domains",
    "register_domain",
    "register_domain_spec",
    "register_gmsh_domain",
    "register_gmsh_planar_domain",
    "sample_coordinate_region",
]
