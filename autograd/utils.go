package main

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"
)

// Read csv file
func ReadCSV(filePath string) [][]string {
	f, err := os.Open(filePath)
	if err != nil {
		log.Fatal("Unable to read input file " + filePath)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatal("Unable to parse file as CSV for " + filePath)
	}
	return records[1:] // Skip header
}

// Load moons data
func LoadMoonsData() ([][]float64, []float64) {
	records := ReadCSV("../make_moons.csv")
	xs := make([][]float64, len(records))
	ys := make([]float64, len(records))

	for i, record := range records {
		xs_0, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			log.Fatal("Unable to convert string to float64 for " + record[0])
		}
		xs_1, err := strconv.ParseFloat(record[1], 64)
		if err != nil {
			log.Fatal("Unable to convert string to float64 for " + record[1])
		}
		xs[i] = []float64{xs_0, xs_1}

		value, err := strconv.ParseFloat(record[2], 64)
		if err != nil {
			log.Fatal("Unable to convert string to float64 for " + record[2])
		}
		ys[i] = value
	}
	return xs, ys
}
