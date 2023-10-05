
def add_plotly(fp):

    with fp.open("r") as f:
        result = f.read()

    with fp.open("w+") as f:

        quarto_include = '<script src="../../../site_libs/quarto-nav/quarto-nav.js"></script>'
        plotly_include = '<script src="https://cdn.plot.ly/plotly-2.9.0.min.js"></script>'
        #
        assert quarto_include in result

        # print(result.count(plotly_include))

        # print(result[:100])
        result = result.replace(
            quarto_include,
            "\n".join(
                [
                    quarto_include,
                    plotly_include,
                ]
            ),
        )

        if result.count(plotly_include) > 1:
            result = result.replace(
                plotly_include,
                "_PLOTLY_",
                2
            )
            result = result.replace(
                plotly_include,
                ""
            )
            result = result.replace(
                "_PLOTLY_",
                plotly_include,
            )
        f.write(result)

    with fp.open("r") as f:
        result = f.read()

    assert plotly_include in result