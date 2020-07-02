package utils;

import java.time.LocalDate;
import java.util.Comparator;

public class CustomComparators {

    static class Employee implements Comparable<Employee> {

        private Long id;
        private String name;
        private LocalDate dob;

        @Override
        public int compareTo(Employee o) {
            return this.id > o.id ? 1 : -1;
        }
    }

    static class NameSorter implements Comparator<Employee> {
        @Override
        public int compare(Employee e1, Employee e2) {
            return e1.name.compareToIgnoreCase(e2.name);
        }
    }

}