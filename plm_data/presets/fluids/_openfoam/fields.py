from pathlib import Path
import textwrap

import numpy as np

from plm_data.core.config import FieldExpressionConfig
from plm_data.core.spatial_fields import is_exact_zero_field_expression

from .core import (
    _format_scalar,
    _format_vector,
    _is_uniform_scalar_expression,
    _is_uniform_vector_expression,
    _scalar_field_cpp_expr,
    _vector_field_cpp_expr,
    _write_foam_header,
    _write_text,
)


def _scalar_fixed_value_block(
    *,
    expr: FieldExpressionConfig,
    parameters: dict[str, float],
    gdim: int,
    field_name: str,
    additive_offset: float = 0.0,
) -> str:
    uniform_value = _is_uniform_scalar_expression(expr, parameters=parameters)
    if uniform_value is not None:
        return textwrap.dedent(
            f"""\
            type            fixedValue;
            value           uniform {_format_scalar(uniform_value + additive_offset)};
            """
        )
    return _coded_fixed_value_scalar_block(
        expr=expr,
        parameters=parameters,
        gdim=gdim,
        field_name=field_name,
        additive_offset=additive_offset,
    )


def _pressure_from_density_temperature_block(
    *,
    density_expr: FieldExpressionConfig,
    temperature_expr: FieldExpressionConfig,
    gas_constant: float,
    parameters: dict[str, float],
    gdim: int,
    field_name: str,
) -> str:
    density_uniform = _is_uniform_scalar_expression(
        density_expr,
        parameters=parameters,
    )
    temperature_uniform = _is_uniform_scalar_expression(
        temperature_expr,
        parameters=parameters,
    )
    if density_uniform is not None and temperature_uniform is not None:
        return textwrap.dedent(
            f"""\
            type            fixedValue;
            value           uniform {_format_scalar(density_uniform * gas_constant * temperature_uniform)};
            """
        )

    density_cpp = _scalar_field_cpp_expr(
        density_expr,
        parameters=parameters,
        gdim=gdim,
    )
    temperature_cpp = _scalar_field_cpp_expr(
        temperature_expr,
        parameters=parameters,
        gdim=gdim,
    )
    pressure_cpp = (
        f"(({density_cpp})*{_format_scalar(gas_constant)}*({temperature_cpp}))"
    )
    return textwrap.dedent(
        f"""\
        type            codedFixedValue;
        name            {field_name}_coded;
        code
        #{{
            const auto& faceCentres = this->patch().Cf();
            scalarField values(faceCentres.size(), 0.0);
            forAll(faceCentres, facei)
            {{
                const scalar x = faceCentres[facei].x();
                const scalar y = faceCentres[facei].y();
                const scalar z = faceCentres[facei].z();
                values[facei] = {pressure_cpp};
            }}
            operator==(values);
        #}};
        value           uniform 0;
        """
    )


def _pressure_from_density_and_patch_temperature_block(
    *,
    density_expr: FieldExpressionConfig,
    gas_constant: float,
    parameters: dict[str, float],
    gdim: int,
    field_name: str,
    temperature_field_name: str = "T",
) -> str:
    density_cpp = _scalar_field_cpp_expr(
        density_expr,
        parameters=parameters,
        gdim=gdim,
    )
    return textwrap.dedent(
        f"""\
        type            codedFixedValue;
        name            {field_name}_coded;
        codeInclude
        #{{
            #include "volFields.H"
        #}};
        code
        #{{
            const auto& faceCentres = this->patch().Cf();
            const auto& temperatureField =
                this->db().lookupObject<Foam::volScalarField>("{temperature_field_name}");
            const auto& temperaturePatch =
                temperatureField.boundaryField()[this->patch().index()];
            scalarField values(faceCentres.size(), 0.0);
            forAll(faceCentres, facei)
            {{
                const scalar x = faceCentres[facei].x();
                const scalar y = faceCentres[facei].y();
                const scalar z = faceCentres[facei].z();
                values[facei] =
                    ({density_cpp})*{_format_scalar(gas_constant)}*temperaturePatch[facei];
            }}
            operator==(values);
        #}};
        value           uniform 0;
        """
    )


def _total_pressure_from_density_temperature_block(
    *,
    density_expr: FieldExpressionConfig,
    temperature_expr: FieldExpressionConfig,
    gas_constant: float,
    parameters: dict[str, float],
    gdim: int,
    field_name: str,
) -> str:
    density_uniform = _is_uniform_scalar_expression(
        density_expr,
        parameters=parameters,
    )
    temperature_uniform = _is_uniform_scalar_expression(
        temperature_expr,
        parameters=parameters,
    )
    if density_uniform is None or temperature_uniform is None:
        return _pressure_from_density_temperature_block(
            density_expr=density_expr,
            temperature_expr=temperature_expr,
            gas_constant=gas_constant,
            parameters=parameters,
            gdim=gdim,
            field_name=field_name,
        )
    total_pressure = density_uniform * gas_constant * temperature_uniform
    return textwrap.dedent(
        f"""\
        type            totalPressure;
        p0              uniform {_format_scalar(total_pressure)};
        """
    )


def _pressure_inlet_outlet_velocity_block(
    *,
    expr: FieldExpressionConfig,
    parameters: dict[str, float],
    gdim: int,
) -> str:
    uniform_value = _is_uniform_vector_expression(
        expr,
        parameters=parameters,
        gdim=gdim,
    )
    if uniform_value is None:
        raise ValueError(
            "OpenFOAM-backed compressible_navier_stokes requires outlet velocity "
            "Dirichlet data to be spatially uniform when translated to "
            "pressureInletOutletVelocity."
        )
    lines = [
        "type            pressureInletOutletVelocity;",
        f"tangentialVelocity constant {_format_vector(uniform_value)};",
        "value           uniform (0 0 0);",
    ]
    return "\n".join(lines)


def _scalar_inlet_outlet_block(
    *,
    expr: FieldExpressionConfig,
    parameters: dict[str, float],
    gdim: int,
    field_name: str,
) -> str:
    uniform_value = _is_uniform_scalar_expression(expr, parameters=parameters)
    if uniform_value is None:
        raise ValueError(
            "OpenFOAM-backed compressible_navier_stokes requires outlet scalar "
            "Dirichlet data to be spatially uniform when translated to "
            "inletOutlet."
        )
    return textwrap.dedent(
        f"""\
        type            inletOutlet;
        inletValue      uniform {_format_scalar(uniform_value)};
        value           uniform {_format_scalar(uniform_value)};
        """
    )


def _vector_fixed_value_block(
    *,
    expr: FieldExpressionConfig,
    parameters: dict[str, float],
    gdim: int,
    field_name: str,
    zero_patch_type: str | None = None,
) -> str:
    uniform_value = _is_uniform_vector_expression(
        expr,
        parameters=parameters,
        gdim=gdim,
    )
    if uniform_value is not None:
        if zero_patch_type is not None and np.allclose(uniform_value, 0.0):
            return f"type            {zero_patch_type};"
        return textwrap.dedent(
            f"""\
            type            fixedValue;
            value           uniform {_format_vector(uniform_value)};
            """
        )
    return _coded_fixed_value_vector_block(
        expr=expr,
        parameters=parameters,
        gdim=gdim,
        field_name=field_name,
    )


def _scaled_scalar_cpp_expr(
    *,
    expr: FieldExpressionConfig,
    parameters: dict[str, float],
    gdim: int,
    gradient_scale: float,
) -> str:
    scalar_expr = _scalar_field_cpp_expr(expr, parameters=parameters, gdim=gdim)
    if gradient_scale == 1.0:
        return scalar_expr
    return f"(({scalar_expr})*{_format_scalar(gradient_scale)})"


def _scaled_vector_cpp_expr(
    *,
    expr: FieldExpressionConfig,
    parameters: dict[str, float],
    gdim: int,
    gradient_scale: float,
) -> str:
    vector_expr = _vector_field_cpp_expr(expr, parameters=parameters, gdim=gdim)
    if gradient_scale == 1.0:
        return vector_expr
    return f"({vector_expr}*{_format_scalar(gradient_scale)})"


def _scalar_fixed_gradient_block(
    *,
    expr: FieldExpressionConfig,
    parameters: dict[str, float],
    gdim: int,
    field_name: str,
    gradient_scale: float = 1.0,
) -> str:
    uniform_value = _is_uniform_scalar_expression(expr, parameters=parameters)
    if uniform_value is not None:
        gradient_value = uniform_value * gradient_scale
        if np.isclose(gradient_value, 0.0):
            return "type            zeroGradient;"
        return textwrap.dedent(
            f"""\
            type            fixedGradient;
            gradient        uniform {_format_scalar(gradient_value)};
            """
        )
    if is_exact_zero_field_expression(expr, parameters):
        return "type            zeroGradient;"
    return _coded_fixed_gradient_scalar_block(
        expr=expr,
        parameters=parameters,
        gdim=gdim,
        field_name=field_name,
        gradient_scale=gradient_scale,
    )


def _vector_fixed_gradient_block(
    *,
    expr: FieldExpressionConfig,
    parameters: dict[str, float],
    gdim: int,
    field_name: str,
    gradient_scale: float = 1.0,
) -> str:
    uniform_value = _is_uniform_vector_expression(
        expr,
        parameters=parameters,
        gdim=gdim,
    )
    if uniform_value is not None:
        gradient_value = uniform_value * gradient_scale
        if np.allclose(gradient_value, 0.0):
            return "type            zeroGradient;"
        return textwrap.dedent(
            f"""\
            type            fixedGradient;
            gradient        uniform {_format_vector(gradient_value)};
            """
        )
    if is_exact_zero_field_expression(expr, parameters):
        return "type            zeroGradient;"
    return _coded_fixed_gradient_vector_block(
        expr=expr,
        parameters=parameters,
        gdim=gdim,
        field_name=field_name,
        gradient_scale=gradient_scale,
    )


def _coded_fixed_value_scalar_block(
    *,
    expr: FieldExpressionConfig,
    parameters: dict[str, float],
    gdim: int,
    field_name: str,
    additive_offset: float = 0.0,
) -> str:
    scalar_expr = _scalar_field_cpp_expr(expr, parameters=parameters, gdim=gdim)
    if additive_offset != 0.0:
        scalar_expr = f"({scalar_expr} + {_format_scalar(additive_offset)})"
    return textwrap.dedent(
        f"""\
        type            codedFixedValue;
        name            {field_name}_coded;
        code
        #{{
            const auto& faceCentres = this->patch().Cf();
            scalarField values(faceCentres.size(), 0.0);
            forAll(faceCentres, facei)
            {{
                const scalar x = faceCentres[facei].x();
                const scalar y = faceCentres[facei].y();
                const scalar z = faceCentres[facei].z();
                values[facei] = {scalar_expr};
            }}
            operator==(values);
        #}};
        value           uniform 0;
        """
    )


def _coded_fixed_value_vector_block(
    *,
    expr: FieldExpressionConfig,
    parameters: dict[str, float],
    gdim: int,
    field_name: str,
) -> str:
    vector_expr = _vector_field_cpp_expr(expr, parameters=parameters, gdim=gdim)
    return textwrap.dedent(
        f"""\
        type            codedFixedValue;
        name            {field_name}_coded;
        code
        #{{
            const auto& faceCentres = this->patch().Cf();
            vectorField values(faceCentres.size(), Foam::vector::zero);
            forAll(faceCentres, facei)
            {{
                const scalar x = faceCentres[facei].x();
                const scalar y = faceCentres[facei].y();
                const scalar z = faceCentres[facei].z();
                values[facei] = {vector_expr};
            }}
            operator==(values);
        #}};
        value           uniform (0 0 0);
        """
    )


def _coded_fixed_gradient_scalar_block(
    *,
    expr: FieldExpressionConfig,
    parameters: dict[str, float],
    gdim: int,
    field_name: str,
    gradient_scale: float = 1.0,
) -> str:
    scalar_expr = _scaled_scalar_cpp_expr(
        expr=expr,
        parameters=parameters,
        gdim=gdim,
        gradient_scale=gradient_scale,
    )
    return textwrap.dedent(
        f"""\
        type            codedMixed;
        refValue        uniform 0;
        refGradient     uniform 0;
        valueFraction   uniform 0;
        name            {field_name}_gradient_coded;
        code
        #{{
            const auto& faceCentres = this->patch().Cf();
            scalarField refValues(faceCentres.size(), 0.0);
            scalarField gradients(faceCentres.size(), 0.0);
            scalarField fractions(faceCentres.size(), 0.0);
            forAll(faceCentres, facei)
            {{
                const scalar x = faceCentres[facei].x();
                const scalar y = faceCentres[facei].y();
                const scalar z = faceCentres[facei].z();
                gradients[facei] = {scalar_expr};
            }}
            this->refValue() = refValues;
            this->refGrad() = gradients;
            this->valueFraction() = fractions;
        #}};
        value           uniform 0;
        """
    )


def _coded_fixed_gradient_vector_block(
    *,
    expr: FieldExpressionConfig,
    parameters: dict[str, float],
    gdim: int,
    field_name: str,
    gradient_scale: float = 1.0,
) -> str:
    vector_expr = _scaled_vector_cpp_expr(
        expr=expr,
        parameters=parameters,
        gdim=gdim,
        gradient_scale=gradient_scale,
    )
    return textwrap.dedent(
        f"""\
        type            codedMixed;
        refValue        uniform (0 0 0);
        refGradient     uniform (0 0 0);
        valueFraction   uniform 0;
        name            {field_name}_gradient_coded;
        code
        #{{
            const auto& faceCentres = this->patch().Cf();
            vectorField refValues(faceCentres.size(), Foam::vector::zero);
            vectorField gradients(faceCentres.size(), Foam::vector::zero);
            scalarField fractions(faceCentres.size(), 0.0);
            forAll(faceCentres, facei)
            {{
                const scalar x = faceCentres[facei].x();
                const scalar y = faceCentres[facei].y();
                const scalar z = faceCentres[facei].z();
                gradients[facei] = {vector_expr};
            }}
            this->refValue() = refValues;
            this->refGrad() = gradients;
            this->valueFraction() = fractions;
        #}};
        value           uniform (0 0 0);
        """
    )


def _render_boundary_field(boundary_entries: dict[str, str]) -> str:
    blocks = []
    for patch_name, body in boundary_entries.items():
        blocks.append(f"{patch_name}\n{{\n{textwrap.indent(body.rstrip(), '    ')}\n}}")
    return "boundaryField\n{\n" + textwrap.indent("\n\n".join(blocks), "    ") + "\n}\n"


def _write_scalar_field_file(
    path: Path,
    *,
    object_name: str,
    dimensions: str,
    internal_values: np.ndarray | None,
    internal_uniform: float | None,
    boundary_entries: dict[str, str],
) -> None:
    if (internal_values is None) == (internal_uniform is None):
        raise ValueError(
            f"Scalar field '{object_name}' must provide exactly one internal value "
            "representation."
        )

    body_lines = [f"dimensions      {dimensions};", ""]
    if internal_uniform is not None:
        body_lines.append(
            f"internalField   uniform {_format_scalar(internal_uniform)};"
        )
    else:
        assert internal_values is not None
        body_lines.extend(
            [
                "internalField   nonuniform List<scalar>",
                str(int(internal_values.shape[0])),
                "(",
                "\n".join(_format_scalar(value) for value in internal_values),
                ");",
            ]
        )
    body_lines.extend(["", _render_boundary_field(boundary_entries)])
    content = (
        _write_foam_header(
            class_name="volScalarField",
            object_name=object_name,
            location="0",
        )
        + "\n".join(body_lines)
        + "\n// ************************************************************************* //\n"
    )
    _write_text(path, content)


def _write_vector_field_file(
    path: Path,
    *,
    object_name: str,
    dimensions: str,
    internal_values: np.ndarray | None,
    internal_uniform: np.ndarray | None,
    boundary_entries: dict[str, str],
) -> None:
    if (internal_values is None) == (internal_uniform is None):
        raise ValueError(
            f"Vector field '{object_name}' must provide exactly one internal value "
            "representation."
        )

    body_lines = [f"dimensions      {dimensions};", ""]
    if internal_uniform is not None:
        body_lines.append(
            f"internalField   uniform {_format_vector(internal_uniform)};"
        )
    else:
        assert internal_values is not None
        body_lines.extend(
            [
                "internalField   nonuniform List<vector>",
                str(int(internal_values.shape[0])),
                "(",
                "\n".join(_format_vector(value) for value in internal_values),
                ");",
            ]
        )
    body_lines.extend(["", _render_boundary_field(boundary_entries)])
    content = (
        _write_foam_header(
            class_name="volVectorField",
            object_name=object_name,
            location="0",
        )
        + "\n".join(body_lines)
        + "\n// ************************************************************************* //\n"
    )
    _write_text(path, content)
