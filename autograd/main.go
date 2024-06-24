package main

import (
	"fmt"
)

func main() {
	// Sine example (Training a neural network to approximate the sine function)
	model := InitMLP(2, []int{16, 16, 1})
	xs, ys := LoadMoonsData()
	iters := 100

	for i := 0; i < iters; i++ {
		totalLoss, accuracy := Loss(model, xs, ys)

		model.ZeroGrad()
		totalLoss.Backward()

		SGD(model, i, iters)
		fmt.Printf("Step %d loss %.5f, accuracy %.5f%%\n", i, totalLoss.data, accuracy*100)
	}

	grid := make([][]string, 0)
	bound := 20
	for y := -bound; y < bound; y++ {
		row := make([]string, 0)
		for x := -bound; x < bound; x++ {
			k := model.Forward([]*Value{
				Init(float64(x) / float64(bound) * 2.0),
				Init(-float64(y) / float64(bound) * 2.0),
			})[0]
			if k.data > 0.0 {
				row = append(row, "*")
			} else {
				row = append(row, ".")
			}
		}
		grid = append(grid, row)
	}

	for _, row := range grid {
		for _, val := range row {
			fmt.Printf("%s ", val)
		}
		fmt.Println()
	}
}
