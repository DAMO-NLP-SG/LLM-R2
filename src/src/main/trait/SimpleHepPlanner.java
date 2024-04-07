package main.trait;

import org.apache.calcite.plan.hep.HepPlanner;
import org.apache.calcite.plan.hep.HepProgram;

public class SimpleHepPlanner extends HepPlanner {

    public SimpleHepPlanner(HepProgram program) {
        super(program);

    }
}
